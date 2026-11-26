# -*- coding: utf-8 -*-
"""并发下的准确率评测：DeepSeek 882 题对齐 + 并发阶梯 + 原生 evaluator 权威判分。

组合目标（用户需求）：
  - query 来源：collect_local_882.py 的 DeepSeek 882 题（(dataset_id, record_index) 严格对齐）
  - 并发阶梯：bench_multi_query.py 的 1→2→4→8 四档并发 + 计时 + 退避重试
  - 判分：复用原生 evaluator（选项匹配 / 数学判分 / LLM judge），非正则粗 hit

与两个源脚本的关键差异：
  - bench_multi_query.py 读 data/ 全量文件、自拼 prompt、只做选项正则 hit、不判 MATH；
  - collect_local_882.py 对齐 882 但只做单并发能力采集；
  - 本脚本把 882 对齐的题目放进并发阶梯，对每次回答用原生 evaluator 判分，
    产出“并发档位 × 准确率 × 吞吐/延迟”的联合结果。

用法：
  python tools/bench_accuracy_concurrency.py --engine splash-qwen38 --dry-run
  python tools/bench_accuracy_concurrency.py --engine splash-qwen38 --limit-per-dataset 5 --concurrency 1,2
  python tools/bench_accuracy_concurrency.py --engine mlxvlm-gemma

前置（缺一显式报错）：
  - results/bench/<ds>/test/DeepSeek-V4.1-Flash/*.json 存在（882 题来源）
  - 本地服务就绪（Splash /ready、MLX-VLM /health model_ready=true）
  - 已 pip install -r requirements.txt（openai/loguru/datasets 等重依赖）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 复用 882 对齐的引擎预设与目标解析（重依赖在其 main() 内懒加载，模块级导入安全）
try:
    from tools.collect_local_882 import (
        ENGINE_PRESETS, build_base_url, resolve_target_config, DS_ORDER, SOURCE_MODEL)
    from tools.bench_common import (
        BenchRequestError, OpenAICompatClient, SystemMonitor, detect_hardware, summarize_repeats)
    from tools.monitor_machine import derive_phase_effective_rates
except ImportError:  # python tools/bench_accuracy_concurrency.py 直跑
    from collect_local_882 import (  # type: ignore
        ENGINE_PRESETS, build_base_url, resolve_target_config, DS_ORDER, SOURCE_MODEL)
    from bench_common import (  # type: ignore
        BenchRequestError, OpenAICompatClient, SystemMonitor, detect_hardware, summarize_repeats)
    from monitor_machine import derive_phase_effective_rates  # type: ignore

DEFAULT_CONCURRENCY = "1,2,4,8"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 纯逻辑（可单测，不依赖重依赖/远端）
# ---------------------------------------------------------------------------

def build_queries(records: Iterable[Any], only: Optional[set] = None,
                  limit_per_dataset: int = 0) -> List[Dict[str, Any]]:
    """把 DeepSeek 记录转成统一 query，按 (DS_ORDER, record_index) 排序。

    limit_per_dataset<=0 表示全量（即 882 题中该数据集全部）。
    """
    order = {ds: i for i, ds in enumerate(DS_ORDER)}
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        ds = r.dataset_id
        if only and ds not in only:
            continue
        buckets.setdefault(ds, []).append({
            "dataset_id": ds,
            "record_index": r.record_index,
            "prompt": r.prompt,
            "ground_truth": r.ground_truth,
            "origin_query": getattr(r, "origin_query", ""),
        })
    queries: List[Dict[str, Any]] = []
    for ds in sorted(buckets, key=lambda d: order.get(d, len(order))):
        items = sorted(buckets[ds], key=lambda q: q["record_index"])
        if limit_per_dataset > 0:
            items = items[: limit_per_dataset]
        queries.extend(items)
    return queries


def parse_concurrency(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def build_queries_from_items(dataset_id: str, items: Iterable[Dict[str, Any]],
                             limit_per_dataset: int = 0) -> List[Dict[str, Any]]:
    """data/ 能力集模式：从原生 evaluator.load_data 返回的 item 构造 query。

    item 必须含 `prompt` 与 `answer`；record_index 按加载顺序编号（非 882 对齐）。
    """
    queries: List[Dict[str, Any]] = []
    n = 0
    for item in items:
        if limit_per_dataset > 0 and n >= limit_per_dataset:
            break
        if isinstance(item, dict) is False:
            continue
        prompt = item.get("prompt")
        if not prompt:
            continue
        queries.append({"dataset_id": dataset_id, "record_index": n,
                        "prompt": str(prompt),
                        "ground_truth": item.get("answer"),
                        "origin_query": item.get("origin_query", "")})
        n += 1
    return queries


def shard_queries(queries: List[Dict[str, Any]], n_shards: int,
                 shard_index: int) -> List[Dict[str, Any]]:
    """round-robin 切片：把 query 列表按下标取模分成 n 份，取第 shard_index 份。

    用于 --shard-across-levels：五档各跑约 1/5 的题，合计覆盖全部 882 题。
    """
    if n_shards <= 0:
        return []
    return queries[shard_index::n_shards]


def load_manifest(path: str) -> List[Dict[str, Any]]:
    """读取抽样清单 JSONL（每行 {"dataset_id", "record_index"}）。"""
    if not path or not os.path.exists(path):
        raise SystemExit(f"manifest 文件不存在：{path}")
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError as exc:
                raise SystemExit(f"manifest 非法行：{line[:80]}") from exc
            entries.append({"dataset_id": e.get("dataset_id"),
                            "record_index": int(e.get("record_index"))})
    return entries


def build_queries_from_manifest(manifest: Iterable[Dict[str, Any]],
                                items_by_dataset: Dict[str, List[Dict[str, Any]]]
                                ) -> List[Dict[str, Any]]:
    """按抽样清单构造 query：用 (dataset_id, record_index) 精确取题。

    越界索引、未知数据集、空 prompt 跳过；输出按 (DS_ORDER, record_index) 排序。
    """
    queries: List[Dict[str, Any]] = []
    for entry in manifest:
        ds = entry.get("dataset_id")
        idx = entry.get("record_index")
        items = items_by_dataset.get(ds, [])
        if idx is None or idx < 0 or idx >= len(items):
            continue
        item = items[idx]
        prompt = item.get("prompt")
        if not prompt:
            continue
        queries.append({"dataset_id": ds, "record_index": idx,
                        "prompt": str(prompt),
                        "ground_truth": item.get("answer"),
                        "origin_query": item.get("origin_query", "")})
    order = {d: i for i, d in enumerate(DS_ORDER)}
    queries.sort(key=lambda q: (order.get(q["dataset_id"], len(order)), q["record_index"]))
    return queries


def clamp_concurrency(ladder: List[int], service_max: Optional[int]
                      ) -> Tuple[List[int], List[int]]:
    """按服务容量钳制并发阶梯，返回 (保留档位, 被跳过档位)。"""
    if service_max is None:
        return list(ladder), []
    kept = [c for c in ladder if c <= service_max]
    skipped = [c for c in ladder if c > service_max]
    return kept, skipped


def make_request_record(query: Dict[str, Any], raw: Dict[str, Any],
                        eval_result: Optional[Dict[str, Any]],
                        mode: str, model_name: Optional[str] = None,
                        attempt_index: int = 0,
                        retry_count: int = 0) -> Dict[str, Any]:
    """单次请求 + 判分结果 -> 统一记录。

    口径：
      - 请求失败：success=False，is_correct=None，fail_reason=客户端错误信息；
      - 请求成功但判分抛错：success=True，is_correct=None，fail_reason="scoring_error"；
      - 判分成功：is_correct 取 bool(eval_result["is_correct"])（None 保持 None）。

    需求字段映射（端侧精品模型测试指标）：
      - sample_id = "{dataset_id}:{record_index}"；eval_name = dataset_id；
      - correct：is_correct 的 int 表示（True->1 / False->0 / None->None）；
      - error：成功为空字符串，失败/判分错误与 fail_reason 同值；
      - prefill_tokens = prompt_tokens - cache_hit_tokens（缓存信息缺失时 = prompt_tokens）；
      - decode_tokens = completion_tokens；
      - reasoning_tokens / cache_status / cache_hit_tokens 由客户端从服务端
        usage 提取（Splash 已验证支持），缺失保持 None，不默认 0。
    """
    success = bool(raw.get("success"))
    is_correct: Optional[bool] = None
    prediction = None
    fail_reason = None
    if success:
        if eval_result is not None:
            v = eval_result.get("is_correct")
            is_correct = None if v is None else bool(v)
            prediction = eval_result.get("prediction")
        else:
            fail_reason = "scoring_error"
    else:
        fail_reason = raw.get("error") or "request_failed"

    prompt_tokens = raw.get("prompt_tokens")
    completion_tokens = raw.get("completion_tokens")
    cache_status = raw.get("cache_status")
    cache_hit_tokens = raw.get("cache_hit_tokens")
    prefill_tokens = None
    if prompt_tokens is not None:
        prefill_tokens = (prompt_tokens - cache_hit_tokens
                          if isinstance(cache_hit_tokens, int) else prompt_tokens)

    return {
        "sample_id": f"{query['dataset_id']}:{query['record_index']}",
        "eval_name": query["dataset_id"],
        "model_name": model_name,
        "dataset_id": query["dataset_id"],
        "record_index": query["record_index"],
        "prompt": query["prompt"],
        "response": raw.get("content"),
        "is_correct": is_correct,
        "correct": (None if is_correct is None else int(is_correct)),
        "prediction": prediction,
        "ground_truth": query["ground_truth"],
        "fail_reason": fail_reason,
        "error": ("" if fail_reason is None else fail_reason),
        "success": success,
        "finish_reason": raw.get("finish_reason"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": raw.get("reasoning_tokens"),
        "cache_status": cache_status,
        "cache_hit_tokens": cache_hit_tokens,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": completion_tokens,
        "prefill_tok_per_s": raw.get("prefill_tok_per_s"),
        "decode_tok_per_s": raw.get("decode_tok_per_s"),
        "prefill_s": raw.get("prefill_s"),
        "decode_s": raw.get("decode_s"),
        "server_wall_s": raw.get("server_wall_s"),
        "request_start_time": raw.get("request_start_time"),
        "first_token_time": raw.get("first_token_time"),
        "response_end_time": raw.get("response_end_time"),
        "timing_source": raw.get("timing_source"),
        "e2e_s": raw.get("e2e_s"),
        "ttft_s": raw.get("ttft_s"),
        "mode": mode,
        "attempt_index": attempt_index,
        "retry_count": retry_count,
    }


def make_workload_event(record: Dict[str, Any], concurrency: int,
                        wall_time: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """把已完成且成功的请求转成监控侧可增量读取的 workload 事件。

    失败请求不进入 token 吞吐，避免把 0 token 的错误放大成“低吞吐”；
    统计仍由 stage summary 的 success/error 字段负责。
    """
    if not record.get("success"):
        return None
    if record.get("prompt_tokens") is None and record.get("completion_tokens") is None:
        return None
    return {
        "wall_time": time.time() if wall_time is None else wall_time,
        "concurrency": concurrency,
        "dataset_id": record.get("dataset_id"),
        "record_index": record.get("record_index"),
        "mode": record.get("mode"),
        "prompt_tokens": record.get("prompt_tokens"),
        "completion_tokens": record.get("completion_tokens"),
        "ttft_s": record.get("ttft_s"),
        "prefill_s": record.get("prefill_s"),
        "decode_s": record.get("decode_s"),
        "e2e_s": record.get("e2e_s"),
    }


def append_workload_event(path: Optional[str], event: Optional[Dict[str, Any]]) -> None:
    """向 monitor_machine 跟随的 JSONL 追加一个事件并立即 flush。"""
    if not path or event is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def per_dataset_accuracy(requests: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """按 dataset_id 聚合判分准确率；未判分（请求失败/判分失败）不计入。"""
    out: Dict[str, Dict[str, Any]] = {}
    for r in requests:
        ds = r.get("dataset_id") or "unknown"
        bucket = out.setdefault(ds, {"n_scored": 0, "n_correct": 0, "accuracy": None})
        if r.get("success") and r.get("is_correct") is not None:
            bucket["n_scored"] += 1
            if r["is_correct"]:
                bucket["n_correct"] += 1
    for bucket in out.values():
        if bucket["n_scored"]:
            bucket["accuracy"] = round(bucket["n_correct"] / bucket["n_scored"], 4)
    return out


def _stats_mean(values: List[Optional[float]]) -> Optional[float]:
    s = summarize_repeats(values)
    return s["mean"]


def compute_stage_summary(
        requests: List[Dict[str, Any]], wall_s: float, concurrency: int,
        theoretical_tflops: Optional[float] = None,
        official_bandwidth_gb_s: Optional[float] = None,
        prefill_flops_per_token: Optional[float] = None,
        decode_flops_per_token: Optional[float] = None,
        prefill_bytes_per_token: Optional[float] = None,
        decode_bytes_per_token: Optional[float] = None,
        flops_per_token: Optional[float] = None,
        bytes_per_token: Optional[float] = None) -> Dict[str, Any]:
    """并发档位汇总：准确率 + 吞吐 + 延迟统计 + 有效算力/带宽。

    Prefill 使用 input_tokens_per_s，Decode 使用 output_tokens_per_s。
    ``flops_per_token`` / ``bytes_per_token`` 是兼容旧命令的通用回退值；
    正式 MoE 评测应显式给分阶段激活参数口径。
    """
    ok = [r for r in requests if r.get("success")]
    scored = [r for r in ok if r.get("is_correct") is not None]
    correct = [r for r in scored if r["is_correct"]]
    decode = [r.get("decode_tok_per_s") for r in ok if r.get("decode_tok_per_s") is not None]
    prefill = [r.get("prefill_tok_per_s") for r in ok if r.get("prefill_tok_per_s") is not None]
    ttft = [r.get("ttft_s") for r in ok if r.get("ttft_s") is not None]
    pt = [r.get("prompt_tokens") for r in ok if r.get("prompt_tokens") is not None]
    ct = [r.get("completion_tokens") for r in ok if r.get("completion_tokens") is not None]
    total_prompt = sum(pt) if pt else None
    total_completion = sum(ct) if ct else None
    tokens_per_s = None
    if wall_s > 0 and (total_prompt is not None or total_completion is not None):
        tokens_per_s = round(((total_prompt or 0) + (total_completion or 0)) / wall_s, 2)
    input_tokens_per_s = (round(total_prompt / wall_s, 2)
                          if wall_s > 0 and total_prompt is not None else None)
    output_tokens_per_s = (round(total_completion / wall_s, 2)
                           if wall_s > 0 and total_completion is not None else None)
    prefill_flops = (prefill_flops_per_token if prefill_flops_per_token is not None
                     else flops_per_token)
    decode_flops = (decode_flops_per_token if decode_flops_per_token is not None
                    else flops_per_token)
    prefill_bytes = (prefill_bytes_per_token if prefill_bytes_per_token is not None
                     else bytes_per_token)
    decode_bytes = (decode_bytes_per_token if decode_bytes_per_token is not None
                    else bytes_per_token)
    compute = derive_phase_effective_rates(
        theoretical_tflops, official_bandwidth_gb_s,
        input_tokens_per_s, output_tokens_per_s,
        prefill_flops, decode_flops, prefill_bytes, decode_bytes)
    summary = {
        "concurrency": concurrency,
        "wall_s": round(wall_s, 3),
        "total_requests": len(requests),
        "success_requests": len(ok),
        "error_requests": len(requests) - len(ok),
        "success_rate": round(len(ok) / len(requests), 4) if requests else None,
        "error_rate": round((len(requests) - len(ok)) / len(requests), 4) if requests else None,
        "n_scored": len(scored),
        "n_correct": len(correct),
        "accuracy": (len(correct) / len(scored)) if scored else None,
        "requests_per_s": round(len(ok) / wall_s, 3) if wall_s > 0 else None,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "tokens_per_s": tokens_per_s,
        "input_tokens_per_s": input_tokens_per_s,
        "output_tokens_per_s": output_tokens_per_s,
        "decode_tok_per_s_mean": round(_stats_mean(decode), 2) if decode else None,
        "prefill_tok_per_s_mean": round(_stats_mean(prefill), 2) if prefill else None,
        "ttft_s_mean": round(_stats_mean(ttft), 4) if ttft else None,
        "ttft_s_p95": round(summarize_repeats(ttft)["p95"], 4) if ttft else None,
        "per_dataset_accuracy": per_dataset_accuracy(requests),
    }
    summary.update(compute)
    summary["compute"] = {
        **compute,
        "measurement": "derived_from_stage_token_throughput",
        "theoretical_tflops": theoretical_tflops,
        "official_bandwidth_gb_s": official_bandwidth_gb_s,
        "prefill_flops_per_token": prefill_flops,
        "decode_flops_per_token": decode_flops,
        "prefill_bytes_per_token": prefill_bytes,
        "decode_bytes_per_token": decode_bytes,
    }
    return summary


# ---------------------------------------------------------------------------
# 单题请求 + 判分
# ---------------------------------------------------------------------------

def should_abort(summary: Dict[str, Any], max_rss_gib: Optional[float],
                 max_swap_gib: Optional[float],
                 max_error_rate: Optional[float]) -> List[str]:
    """判断某并发档是否应熔断；返回触发原因列表（空=继续升档）。

    任一指标为 None（未采集/未设置阈值）都不触发；阈值设为 None 表示关闭该熔断项。
    """
    reasons: List[str] = []
    mem = summary.get("memory") or {}
    rss = mem.get("peak_process_gib")
    swap = mem.get("swap_delta_gib")
    err = summary.get("error_rate")
    if max_rss_gib is not None and rss is not None and rss > max_rss_gib:
        reasons.append(f"rss_peak {rss:.2f}GiB > {max_rss_gib}GiB")
    if max_swap_gib is not None and swap is not None and swap > max_swap_gib:
        reasons.append(f"swap_delta {swap:.3f}GiB > {max_swap_gib}GiB")
    if max_error_rate is not None and err is not None and err > max_error_rate:
        reasons.append(f"error_rate {err:.2%} > {max_error_rate:.2%}")
    return reasons


def _score(evaluator: Any, query: Dict[str, Any], output_text: Optional[str]):
    """调用原生 evaluator 判分；返回 (is_correct, prediction)。"""
    if output_text is None or output_text == "":
        return {"is_correct": False, "prediction": None}  # 无回答按错误计
    return evaluator.evaluate({"answer": query["ground_truth"]}, output_text)


def run_one(client: OpenAICompatClient, query: Dict[str, Any], evaluator: Any,
            mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    """单 query 带退避重试的请求 + 判分。"""
    max_attempts = max(1, int(args.max_attempts))
    backoff_base = max(0.0, float(args.backoff_base_s))
    messages = [{"role": "user", "content": query["prompt"]}]
    for attempt in range(max_attempts):
        try:
            if mode == "stream":
                raw = client.chat_stream(messages=messages, max_tokens=args.max_output_tokens,
                                         temperature=args.temperature, top_p=args.top_p,
                                         seed=args.seed)
            else:
                raw = client.chat_nonstream(messages=messages, max_tokens=args.max_output_tokens,
                                            temperature=args.temperature, top_p=args.top_p,
                                            seed=args.seed)
            eval_result = None
            if raw.get("success"):
                try:
                    eval_result = _score(evaluator, query, raw.get("content"))
                except Exception:
                    eval_result = None
            record = make_request_record(query, raw, eval_result, mode,
                                         model_name=args.engine,
                                         attempt_index=attempt, retry_count=attempt)
            if record["success"]:
                return record
        except BenchRequestError as exc:
            raw = {"success": False, "content": None, "error": str(exc)}
            record = make_request_record(query, raw, None, mode,
                                         model_name=args.engine,
                                         attempt_index=attempt, retry_count=attempt)
        if attempt < max_attempts - 1:
            time.sleep(min(30.0, backoff_base * (2 ** attempt)))
    return record


def run_stage(concurrency: int, queries: List[Dict[str, Any]], client: OpenAICompatClient,
              evaluators: Dict[str, Any], args: argparse.Namespace,
              workload_counters_path: Optional[str] = None
              ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """一个并发档位：warmup + 对全部 query 并发请求判分。"""
    modes: List[str] = []
    if args.mode in ("nonstream", "both"):
        modes.append("nonstream")
    if args.mode in ("stream", "both"):
        modes.append("stream")

    # 预热：首 min(concurrency, len) 题各跑一遍（不判分统计）
    if queries and args.warmup > 0:
        warm_q = queries[: min(concurrency, len(queries))]
        warm_tasks = [(q, m) for _ in range(args.warmup) for q in warm_q for m in modes]
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(ex.map(lambda it: run_one(client, it[0], evaluators[it[0]["dataset_id"]],
                                           it[1], args), warm_tasks))

    tasks = [(q, m) for _ in range(args.repeat) for q in queries for m in modes]
    monitor = SystemMonitor(pid=getattr(args, "server_pid", None),
                            disk=getattr(args, "monitor_disk", None))
    monitor.start()
    started = time.perf_counter()
    requests: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(run_one, client, q, evaluators[q["dataset_id"]], m, args)
                for q, m in tasks]
        for fut in as_completed(futs):
            try:
                record = fut.result()
            except Exception as exc:  # 防御性兜底
                record = {"dataset_id": None, "record_index": None, "success": False,
                          "is_correct": None, "fail_reason": f"future_error: {exc}",
                          "mode": None, "ttft_s": None, "decode_tok_per_s": None,
                          "prefill_tok_per_s": None, "e2e_s": None}
            requests.append(record)
            append_workload_event(workload_counters_path,
                                  make_workload_event(record, concurrency))
    wall_s = time.perf_counter() - started
    memory, ssd_read = monitor.stop()
    summary = compute_stage_summary(
        requests, wall_s, concurrency,
        theoretical_tflops=getattr(args, "theoretical_tflops", None),
        official_bandwidth_gb_s=getattr(args, "official_bandwidth_gb_s", None),
        prefill_flops_per_token=getattr(args, "prefill_flops_per_token", None),
        decode_flops_per_token=getattr(args, "decode_flops_per_token", None),
        prefill_bytes_per_token=getattr(args, "prefill_bytes_per_token", None),
        decode_bytes_per_token=getattr(args, "decode_bytes_per_token", None),
        flops_per_token=getattr(args, "flops_per_token", None),
        bytes_per_token=getattr(args, "bytes_per_token", None))
    summary["memory"] = memory
    summary["ssd_read_mib_s"] = ssd_read
    return summary, requests


# ---------------------------------------------------------------------------
# 文件输出
# ---------------------------------------------------------------------------

def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "model"


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def render_markdown(stages: List[Dict[str, Any]], query_count: int) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 并发下的准确率评测汇总\n",
        f"> 生成时间：{now}；query 总数（对齐 DeepSeek 882）：{query_count}\n",
        "| 并发 | 准确率 | n_scored | 成功请求 | 成功率 | 吞吐 req/s | "
        "decode tok/s | TTFT mean (s) | 错误数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stages:
        acc = f"{s['accuracy'] * 100:.1f}%" if s.get("accuracy") is not None else "-"
        ttft = f"{s['ttft_s_mean']:.3f}" if s.get("ttft_s_mean") is not None else "-"
        lines.append(
            f"| {s['concurrency']} | {acc} | {s['n_scored']} | {s['success_requests']} "
            f"| {s.get('success_rate') or '-'} | {s.get('requests_per_s') or '-'} "
            f"| {s.get('decode_tok_per_s_mean') or '-'} | {ttft} | {s['error_requests']} |")
    lines += [
        "",
        "## 有效算力与有效带宽（workload-level 推导值）",
        "",
        "> 这些值由完成请求的 token 吞吐和显式指定的每 token FLOPs/字节数推导，"
        "不是 Apple GPU/ANE 物理计数器实测值；缺少参数时保持空值。"
        "Prefill 使用输入 token 吞吐，Decode 使用输出 token 吞吐。",
        "",
        "| 并发 | input tok/s | output tok/s | Prefill TFLOPS | Decode TFLOPS | "
        "总有效带宽 GB/s | TFLOPS 利用率 | 带宽利用率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stages:
        def fmt(v, nd=3):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"
        lines.append(
            f"| {s['concurrency']} | {fmt(s.get('input_tokens_per_s'), 2)} "
            f"| {fmt(s.get('output_tokens_per_s'), 2)} "
            f"| {fmt(s.get('prefill_effective_tflops'))} "
            f"| {fmt(s.get('decode_effective_tflops'))} "
            f"| {fmt(s.get('effective_bandwidth_gb_s'))} "
            f"| {fmt(s.get('tflops_utilization'), 4)} "
            f"| {fmt(s.get('bandwidth_utilization'), 4)} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_accuracy_concurrency",
        description="并发阶梯下对 DeepSeek 882 题做原生 evaluator 判分的准确率评测")
    p.add_argument("--engine", required=True,
                   choices=list(ENGINE_PRESETS) + ["custom"],
                   help="本地引擎预设；custom 需给 --base-url/--api-model/--target-model")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--api-model", default=None, help="API 模型 ID，覆盖预设")
    p.add_argument("--target-model", default=None, help="结果目录模型名，覆盖预设")
    p.add_argument("--base-url", default=None, help="完整 base_url 覆盖（含 /v1）")
    p.add_argument("--engine-version", default=None)
    p.add_argument("--weight-format", default=None, help="gguf / mlx")
    p.add_argument("--quantization", default=None)
    p.add_argument("--draft-model", default=None)
    p.add_argument("--kv-format", default=None)
    p.add_argument("--state-format", default=None)
    p.add_argument("--model-revision-or-digest", default=None)
    p.add_argument("--query-source", choices=("data", "882", "manifest"), default="data",
                   help="query 来源：data=原生能力集全量（无需 results/bench）；"
                        "882=对齐 DeepSeek 882 题（需 results/bench）")
    p.add_argument("--manifest", default=None,
                   help="抽样清单 JSONL（query-source=manifest 时必填）")
    p.add_argument("--shard-across-levels", action="store_true",
                   help="把 query 按并发档数 round-robin 切片，每档只跑自己的那份，"
                        "合计覆盖全部题目（省时）")
    p.add_argument("--service-max-concurrency", type=int, default=None,
                   help="服务端 queue-size/max-num-seqs 上限；超过的并发档跳过")
    p.add_argument("--max-rss-gib", type=float, default=55.0,
                   help="服务进程峰值 RSS 熔断阈值（GiB），默认 55")
    p.add_argument("--max-swap-gib", type=float, default=4.0,
                   help="swap 增量熔断阈值（GiB），默认 4")
    p.add_argument("--max-error-rate", type=float, default=0.3,
                   help="错误率熔断阈值（0~1），默认 0.3")
    p.add_argument("--concurrency", type=parse_concurrency, default=parse_concurrency(DEFAULT_CONCURRENCY),
                   help="并发阶梯，默认 1,2,4,8（四档，最大并发 8）")
    p.add_argument("--repeat", type=int, default=1, help="每个 query 每 mode 重复次数")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--backoff-base-s", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--mode", choices=("stream", "nonstream", "both"), default="nonstream")
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--cold-load-s", type=float, default=None,
                   help="冷加载秒数（服务进程启动 -> /ready，启动方测量后传入）")
    p.add_argument("--datasets", default="", help="逗号分隔 dataset_id，默认全部 6 类（882 对齐子集）")
    p.add_argument("--limit-per-dataset", type=int, default=0,
                   help="每数据集最多采几题（0=全部 882 对齐题），用于冒烟")
    p.add_argument("--server-pid", type=int, default=None)
    p.add_argument("--monitor-disk", default=None)
    p.add_argument("--live-counters-file", default=None,
                   help="每个请求完成时追加 workload JSONL，供 monitor_machine 实时跟随")
    p.add_argument("--workload-window-s", type=float, default=15.0,
                   help="监控侧实时 token/s 滑动窗口秒数（写入 manifest）")
    p.add_argument("--theoretical-tflops", type=float, default=None,
                   help="硬件理论峰值算力 TFLOPS，仅作利用率分母")
    p.add_argument("--official-bandwidth-gb-s", type=float, default=None,
                   help="硬件理论统一内存带宽 GB/s，仅作利用率分母")
    p.add_argument("--flops-per-token", type=float, default=None,
                   help="通用每 token FLOPs；分阶段参数缺省时回退使用")
    p.add_argument("--bytes-per-token", type=float, default=None,
                   help="通用每 token 字节数；分阶段参数缺省时回退使用")
    p.add_argument("--prefill-flops-per-token", type=float, default=None)
    p.add_argument("--decode-flops-per-token", type=float, default=None)
    p.add_argument("--prefill-bytes-per-token", type=float, default=None)
    p.add_argument("--decode-bytes-per-token", type=float, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true", help="只加载 882 题并打印统计，不请求服务")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    target = resolve_target_config(args)

    # 并发阶梯按服务容量钳制（Splash --queue-size / MLX-VLM --max-num-seqs）
    ladder, skipped = clamp_concurrency(args.concurrency, args.service_max_concurrency)
    if skipped:
        print(f"[warn] 并发档位 {skipped} 超过服务容量 {args.service_max_concurrency}，"
              f"已跳过；实际阶梯={ladder}")
    args.concurrency = ladder

    from evaluation.factory import EvaluatorFactory

    evaluator_factory = EvaluatorFactory()
    only = {d.strip() for d in args.datasets.split(",") if d.strip()} or None
    ds_list = [d for d in DS_ORDER if (only is None or d in only)]

    queries: List[Dict[str, Any]] = []
    evaluators: Dict[str, Any] = {}
    if args.query_source == "data":
        # 原生能力集全量模式：无需 results/bench；用 evaluator.load_data 的 prompt + 原生判分
        for ds in ds_list:
            try:
                evaluator = evaluator_factory.get_evaluator(ds)
                evaluators[ds] = evaluator
                items = evaluator.load_data("test")
            except Exception as exc:
                raise SystemExit(f"{ds}: 加载评测集失败（缺数据或依赖？）: {exc}") from exc
            queries.extend(build_queries_from_items(ds, items, args.limit_per_dataset))
        src_label = "data/ 原生能力集（非 882 对齐）"
    elif args.query_source == "882":
        # 882 对齐模式：以 DeepSeek 历史矩阵的 prompt/index 为准，进路由矩阵
        from baselines.data_loader import BaselineDataLoader
        loader = BaselineDataLoader(config_path=os.path.join(ROOT, "config",
                                                            "baseline_config_edge.yaml"))
        records = [r for r in loader.load_all_records() if r.model_name == SOURCE_MODEL]
        if not records:
            raise SystemExit(
                f"未找到 {SOURCE_MODEL} 的 882 题来源记录：请确认 results/bench/<ds>/test/"
                f"{SOURCE_MODEL}/ 存在且 config/baseline_config_edge.yaml 指向正确")
        queries = build_queries(records, only=only, limit_per_dataset=args.limit_per_dataset)
        for ds in {q["dataset_id"] for q in queries}:
            evaluators[ds] = evaluator_factory.get_evaluator(ds)
        src_label = f"{SOURCE_MODEL} 882 题对齐"
    else:
        # manifest 模式：按用户抽样清单取题（所有模型共用同一份清单）
        if not args.manifest:
            raise SystemExit("--query-source manifest 必须同时提供 --manifest 路径")
        manifest = load_manifest(args.manifest)
        if not manifest:
            raise SystemExit(f"manifest 为空：{args.manifest}")
        needed = [d for d in DS_ORDER
                  if d in {e.get("dataset_id") for e in manifest}]
        items_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
        for ds in needed:
            try:
                evaluator = evaluator_factory.get_evaluator(ds)
                evaluators[ds] = evaluator
                items_by_dataset[ds] = list(evaluator.load_data("test"))
            except Exception as exc:
                raise SystemExit(f"{ds}: 加载评测集失败: {exc}") from exc
        queries = build_queries_from_manifest(manifest, items_by_dataset)
        src_label = f"manifest 抽样清单（{len(manifest)} 条，{len(queries)} 有效）"

    if not queries:
        raise SystemExit("所选数据集没有可用 query")
    print(f"[load] {len(queries)} query / {src_label}")
    print(f"[target] {target['target_model']} @ {target['base_url']}")
    print(f"[ladder] concurrency={args.concurrency} mode={args.mode} "
          f"repeat={args.repeat}")

    tag = _safe_name(target["target_model"])
    out_dir = args.out or os.path.join("results", "mac_benchmark", "accuracy_concurrency", tag)
    if args.dry_run:
        per_ds: Dict[str, int] = {}
        for q in queries:
            per_ds[q["dataset_id"]] = per_ds.get(q["dataset_id"], 0) + 1
        print(f"[dry-run] per-dataset={per_ds} out={out_dir}")
        return 0

    client = _make_client(target["base_url"], target["api_model"], args.timeout)

    stages: List[Dict[str, Any]] = []
    abort_info: Optional[Dict[str, Any]] = None
    live_counters_file = args.live_counters_file
    if live_counters_file:
        live_counters_file = (live_counters_file if os.path.isabs(live_counters_file)
                              else os.path.join(out_dir, live_counters_file))
        os.makedirs(os.path.dirname(live_counters_file) or ".", exist_ok=True)
        if os.path.exists(live_counters_file):
            os.remove(live_counters_file)
        print(f"[monitor] live counters -> {live_counters_file}", flush=True)
    n_levels = len(args.concurrency)
    for level_idx, concurrency in enumerate(args.concurrency):
        stage_queries = queries
        if args.shard_across_levels:
            stage_queries = shard_queries(queries, n_levels, level_idx)
        print(f"[stage] concurrency={concurrency} queries={len(stage_queries)}", flush=True)
        summary, requests = run_stage(concurrency, stage_queries, client, evaluators, args,
                                      workload_counters_path=live_counters_file)
        stages.append(summary)
        write_json(os.path.join(out_dir, f"stage_{concurrency}.json"), summary)
        write_jsonl(os.path.join(out_dir, f"stage_{concurrency}_requests.jsonl"), requests)
        acc = summary.get("accuracy")
        acc_s = f"{acc * 100:.1f}%" if acc is not None else "-"
        print(f"  acc={acc_s} scored={summary['n_scored']} "
              f"err={summary['error_requests']} "
              f"decode_mean={summary.get('decode_tok_per_s_mean')}", flush=True)
        reasons = should_abort(summary, args.max_rss_gib, args.max_swap_gib,
                               args.max_error_rate)
        if reasons:
            prev = stages[-2]["concurrency"] if len(stages) >= 2 else None
            abort_info = {"aborted_at_concurrency": concurrency,
                          "abort_reasons": reasons,
                          "stable_max_concurrency": prev}
            print(f"[abort] concurrency={concurrency} 触发熔断：{reasons}；"
                  f"稳定最大并发={prev}", flush=True)
            break

    manifest = {
        "target_model": target["target_model"],
        "api_model": target["api_model"],
        "base_url": target["base_url"],
        "engine": args.engine,
        "engine_version": args.engine_version,
        "weight_format": args.weight_format,
        "quantization": args.quantization,
        "draft_model": args.draft_model,
        "kv_format": args.kv_format,
        "state_format": args.state_format,
        "query_source": args.query_source,
        "aligned_to_882": args.query_source == "882",
        "service_max_concurrency": args.service_max_concurrency,
        "max_rss_gib": args.max_rss_gib,
        "max_swap_gib": args.max_swap_gib,
        "max_error_rate": args.max_error_rate,
        "aborted_at_concurrency": abort_info["aborted_at_concurrency"] if abort_info else None,
        "abort_reasons": abort_info["abort_reasons"] if abort_info else [],
        "stable_max_concurrency": (abort_info["stable_max_concurrency"] if abort_info
                                   else (args.concurrency[-1] if args.concurrency else None)),
        "concurrency": args.concurrency,
        "repeat": args.repeat,
        "mode": args.mode,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "cold_load_s": args.cold_load_s,
        "live_counters_file": live_counters_file,
        "workload_window_s": args.workload_window_s,
        "theoretical_tflops": args.theoretical_tflops,
        "official_bandwidth_gb_s": args.official_bandwidth_gb_s,
        "flops_per_token": args.flops_per_token,
        "bytes_per_token": args.bytes_per_token,
        "prefill_flops_per_token": args.prefill_flops_per_token,
        "decode_flops_per_token": args.decode_flops_per_token,
        "prefill_bytes_per_token": args.prefill_bytes_per_token,
        "decode_bytes_per_token": args.decode_bytes_per_token,
        "query_count": len(queries),
        "hardware": detect_hardware(),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(stages, len(queries)))
    print(f"[done] out={out_dir}")
    return 0


def _make_client(base_url: str, model: str, timeout: float) -> OpenAICompatClient:
    """由绝对 base_url 构造客户端（复用 OpenAICompatClient，但 URL 为绝对路径）。"""
    import urllib.parse
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return OpenAICompatClient(host=host, port=port, model=model, timeout_s=timeout)


if __name__ == "__main__":
    raise SystemExit(main())
