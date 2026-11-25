# -*- coding: utf-8 -*-
"""并发下的准确率评测：DeepSeek 882 题对齐 + 并发阶梯 + 原生 evaluator 权威判分。

组合目标（用户需求）：
  - query 来源：collect_local_882.py 的 DeepSeek 882 题（(dataset_id, record_index) 严格对齐）
  - 并发阶梯：bench_multi_query.py 的 1→2→4→… 并发 + 计时 + 退避重试
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
except ImportError:  # python tools/bench_accuracy_concurrency.py 直跑
    from collect_local_882 import (  # type: ignore
        ENGINE_PRESETS, build_base_url, resolve_target_config, DS_ORDER, SOURCE_MODEL)
    from bench_common import (  # type: ignore
        BenchRequestError, OpenAICompatClient, SystemMonitor, detect_hardware, summarize_repeats)

DEFAULT_CONCURRENCY = "1,2,4"


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


def make_request_record(query: Dict[str, Any], raw: Dict[str, Any],
                        eval_result: Optional[Dict[str, Any]],
                        mode: str, attempt_index: int = 0,
                        retry_count: int = 0) -> Dict[str, Any]:
    """单次请求 + 判分结果 -> 统一记录。

    口径：
      - 请求失败：success=False，is_correct=None，fail_reason=客户端错误信息；
      - 请求成功但判分抛错：success=True，is_correct=None，fail_reason="scoring_error"；
      - 判分成功：is_correct 取 bool(eval_result["is_correct"])（None 保持 None）。
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
    return {
        "dataset_id": query["dataset_id"],
        "record_index": query["record_index"],
        "prompt": query["prompt"],
        "response": raw.get("content"),
        "is_correct": is_correct,
        "prediction": prediction,
        "ground_truth": query["ground_truth"],
        "fail_reason": fail_reason,
        "success": success,
        "finish_reason": raw.get("finish_reason"),
        "prompt_tokens": raw.get("prompt_tokens"),
        "completion_tokens": raw.get("completion_tokens"),
        "prefill_tok_per_s": raw.get("prefill_tok_per_s"),
        "decode_tok_per_s": raw.get("decode_tok_per_s"),
        "e2e_s": raw.get("e2e_s"),
        "ttft_s": raw.get("ttft_s"),
        "mode": mode,
        "attempt_index": attempt_index,
        "retry_count": retry_count,
    }


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


def compute_stage_summary(requests: List[Dict[str, Any]], wall_s: float,
                          concurrency: int) -> Dict[str, Any]:
    """并发档位汇总：准确率 + 吞吐 + 延迟统计 + 每数据集正确率。"""
    ok = [r for r in requests if r.get("success")]
    scored = [r for r in ok if r.get("is_correct") is not None]
    correct = [r for r in scored if r["is_correct"]]
    decode = [r.get("decode_tok_per_s") for r in ok if r.get("decode_tok_per_s") is not None]
    prefill = [r.get("prefill_tok_per_s") for r in ok if r.get("prefill_tok_per_s") is not None]
    ttft = [r.get("ttft_s") for r in ok if r.get("ttft_s") is not None]
    pt = [r.get("prompt_tokens") for r in ok if r.get("prompt_tokens") is not None]
    ct = [r.get("completion_tokens") for r in ok if r.get("completion_tokens") is not None]
    return {
        "concurrency": concurrency,
        "wall_s": round(wall_s, 3),
        "total_requests": len(requests),
        "success_requests": len(ok),
        "error_requests": len(requests) - len(ok),
        "success_rate": round(len(ok) / len(requests), 4) if requests else None,
        "n_scored": len(scored),
        "n_correct": len(correct),
        "accuracy": (len(correct) / len(scored)) if scored else None,
        "requests_per_s": round(len(ok) / wall_s, 3) if wall_s > 0 else None,
        "total_prompt_tokens": sum(pt) if pt else None,
        "total_completion_tokens": sum(ct) if ct else None,
        "tokens_per_s": round((sum(pt) + sum(ct)) / wall_s, 2)
                        if wall_s > 0 and (pt or ct) else None,
        "decode_tok_per_s_mean": round(_stats_mean(decode), 2) if decode else None,
        "prefill_tok_per_s_mean": round(_stats_mean(prefill), 2) if prefill else None,
        "ttft_s_mean": round(_stats_mean(ttft), 4) if ttft else None,
        "ttft_s_p95": round(summarize_repeats(ttft)["p95"], 4) if ttft else None,
        "per_dataset_accuracy": per_dataset_accuracy(requests),
    }


# ---------------------------------------------------------------------------
# 单题请求 + 判分
# ---------------------------------------------------------------------------

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
                                         attempt_index=attempt, retry_count=attempt)
            if record["success"]:
                return record
        except BenchRequestError as exc:
            raw = {"success": False, "content": None, "error": str(exc)}
            record = make_request_record(query, raw, None, mode,
                                         attempt_index=attempt, retry_count=attempt)
        if attempt < max_attempts - 1:
            time.sleep(min(30.0, backoff_base * (2 ** attempt)))
    return record


def run_stage(concurrency: int, queries: List[Dict[str, Any]], client: OpenAICompatClient,
              evaluators: Dict[str, Any], args: argparse.Namespace
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
                requests.append(fut.result())
            except Exception as exc:  # 防御性兜底
                requests.append({"dataset_id": None, "record_index": None, "success": False,
                                 "is_correct": None, "fail_reason": f"future_error: {exc}",
                                 "mode": None, "ttft_s": None, "decode_tok_per_s": None,
                                 "prefill_tok_per_s": None, "e2e_s": None})
    wall_s = time.perf_counter() - started
    memory, ssd_read = monitor.stop()
    summary = compute_stage_summary(requests, wall_s, concurrency)
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
    p.add_argument("--concurrency", type=parse_concurrency, default=parse_concurrency(DEFAULT_CONCURRENCY),
                   help="并发阶梯，默认 1,2,4")
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
    p.add_argument("--datasets", default="", help="逗号分隔 dataset_id，默认全部 6 类（882 对齐子集）")
    p.add_argument("--limit-per-dataset", type=int, default=0,
                   help="每数据集最多采几题（0=全部 882 对齐题），用于冒烟")
    p.add_argument("--server-pid", type=int, default=None)
    p.add_argument("--monitor-disk", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true", help="只加载 882 题并打印统计，不请求服务")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    target = resolve_target_config(args)

    # 重依赖懒加载：882 来源 + 原生 evaluator
    from baselines.data_loader import BaselineDataLoader
    from evaluation.factory import EvaluatorFactory

    loader = BaselineDataLoader(
        config_path=str(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "config", "baseline_config_edge.yaml")))
    records = [r for r in loader.load_all_records() if r.model_name == SOURCE_MODEL]
    if not records:
        raise SystemExit(
            f"未找到 {SOURCE_MODEL} 的 882 题来源记录：请确认 results/bench/<ds>/test/"
            f"{SOURCE_MODEL}/ 存在且 config/baseline_config_edge.yaml 指向正确")

    only = {d.strip() for d in args.datasets.split(",") if d.strip()} or None
    queries = build_queries(records, only=only, limit_per_dataset=args.limit_per_dataset)
    if not queries:
        raise SystemExit("所选数据集没有可用 query")
    print(f"[load] {len(queries)} query / {SOURCE_MODEL} 对齐")
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

    evaluator_factory = EvaluatorFactory()
    evaluators: Dict[str, Any] = {}
    for ds in {q["dataset_id"] for q in queries}:
        evaluators[ds] = evaluator_factory.get_evaluator(ds)

    client = _make_client(target["base_url"], target["api_model"], args.timeout)

    stages: List[Dict[str, Any]] = []
    for concurrency in args.concurrency:
        print(f"[stage] concurrency={concurrency}", flush=True)
        summary, requests = run_stage(concurrency, queries, client, evaluators, args)
        stages.append(summary)
        write_json(os.path.join(out_dir, f"stage_{concurrency}.json"), summary)
        write_jsonl(os.path.join(out_dir, f"stage_{concurrency}_requests.jsonl"), requests)
        acc = summary.get("accuracy")
        acc_s = f"{acc * 100:.1f}%" if acc is not None else "-"
        print(f"  acc={acc_s} scored={summary['n_scored']} "
              f"err={summary['error_requests']} "
              f"decode_mean={summary.get('decode_tok_per_s_mean')}", flush=True)

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
        "concurrency": args.concurrency,
        "repeat": args.repeat,
        "mode": args.mode,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
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
