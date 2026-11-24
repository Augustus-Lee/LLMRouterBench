# -*- coding: utf-8 -*-
"""能力评测集并发阶梯多 query 性能测量。

设计目标
--------
1. 使用仓库原生能力评测集（GPQA / MMLU / MMLU-Pro / MATH500 / ARC-C / GSM8K）
   作为统一 query 来源，不再依赖性能 workload；
2. 按并发阶梯 `1 -> 2 -> 4 -> 8 -> 16` 上涨；
3. 每个档位记录 per-request TTFT / E2E / completion tokens / finish_reason
   以及档位吞吐、成功 token 吞吐、p95 等汇总值；
4. 同步采样推理进程 RSS / wired / swap / SSD 读（可选 psutil / macOS）。

用法示例
--------
    python tools/bench_multi_query.py --model Qwen/Qwen3.8-27B --port 18080
    python tools/bench_multi_query.py --model mlx-community/Qwen3.6-35B-A3B-4bit \\
        --port 18081 --limit-per-dataset 0 --concurrency 1,2,4,8,16
    python tools/bench_multi_query.py --dry-run

输出目录：``results/mac_benchmark/multi_query/<model_tag>/``。
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import re
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # 支持包导入与脚本直跑两种方式
    from tools.bench_common import BenchRequestError, OpenAICompatClient, SystemMonitor, detect_hardware
except ImportError:  # python tools/bench_multi_query.py 直跑
    from bench_common import BenchRequestError, OpenAICompatClient, SystemMonitor, detect_hardware  # type: ignore


# ---------------------------------------------------------------------------
# 能力评测集定义
# ---------------------------------------------------------------------------

DEFAULT_DATASETS = ("gpqa", "mmlupro", "math500", "arcc", "gsm8k", "mmlu")

DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "gpqa": {
        "path": "data/GPQA/gpqa_diamond.json",
        "answer_key": "answer",
        "kind": "mcq",
        "question_key": "Question",
        "options_key": "options",
    },
    "mmlupro": {
        "path": "data/MMLUPro/test.json",
        "answer_key": "answer",
        "kind": "mcq",
        "question_key": "question",
        "options_key": "options",
    },
    "math500": {
        "path": "data/MATH500/test.json",
        "answer_key": "answer",
        "kind": "math",
        "question_key": "problem",
        "options_key": None,
    },
    "arcc": {
        "path": "data/arc_c/arc_test.json",
        "answer_key": "answerKey",
        "kind": "mcq",
        "question_key": "question",
        "options_key": "choices",
    },
    "gsm8k": {
        "path": "data/GSM8K/test.json",
        "answer_key": "answer",
        "kind": "numeric",
        "question_key": "question",
        "options_key": None,
    },
    "mmlu": {
        "path": "data/MMLU/test.json",
        "answer_key": "answer",
        "kind": "mcq",
        "question_key": "question",
        "options_key": "options",
    },
}


def _safe_literal(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    stripped = raw.strip()
    if not stripped:
        return raw
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return raw


def _format_options(options: Any) -> str:
    options = _safe_literal(options)
    if isinstance(options, dict):
        if "text" in options:
            options = options["text"]
        else:
            lines = [f"{str(k)}) {str(v)}" for k, v in options.items()]
            return "\n".join(lines)
    if isinstance(options, list):
        return "\n".join(f"{chr(ord('A') + i)}) {item}" for i, item in enumerate(options))
    return str(options)


def build_prompt(dataset_id: str, record: Dict[str, Any]) -> Tuple[str, str]:
    """按数据集构造统一 prompt，并返回预期答案字符串。"""
    spec = DATASET_SPECS[dataset_id]
    question = str(record.get(spec["question_key"], "")).strip()
    expected = str(record.get(spec["answer_key"])).strip() if record.get(spec["answer_key"]) is not None else ""
    options = spec.get("options_key")
    if options and record.get(options):
        formatted = _format_options(record[options])
        prompt = f"{question}\n\nOptions:\n{formatted}\n\nAnswer with the option letter only."
    elif dataset_id == "math500":
        prompt = f"{question}\n\nProvide the final answer."
    elif dataset_id == "gsm8k":
        prompt = f"{question}\n\nProvide the numeric answer."
    else:
        prompt = question
    return prompt, expected


def load_queries(datasets: Iterable[str], limit_per_dataset: int) -> List[Dict[str, Any]]:
    """读取能力评测集 JSONL；返回统一 query 记录。

    `limit_per_dataset <= 0` 表示全量。文件必须逐行 JSON。
    """
    queries: List[Dict[str, Any]] = []
    for dataset_id in datasets:
        spec = DATASET_SPECS[dataset_id]
        path = spec["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{dataset_id}: missing {path}")
        with open(path, "r", encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{row_idx}: invalid JSON: {exc}") from exc
                if limit_per_dataset > 0 and row_idx >= limit_per_dataset:
                    break
                prompt, expected = build_prompt(dataset_id, record)
                if not prompt:
                    continue
                sample_id = record.get("id") or record.get("unique_id") or f"{dataset_id}:{row_idx}"
                queries.append({
                    "sample_id": str(sample_id),
                    "dataset_id": dataset_id,
                    "record_index": row_idx,
                    "prompt": prompt,
                    "expected_answer": expected,
                    "metadata": record,
                })
    if not queries:
        raise ValueError("no queries loaded from selected datasets")
    return queries


# ---------------------------------------------------------------------------
# 请求与判分
# ---------------------------------------------------------------------------

_MC_ANSWER_RE = re.compile(r"\b([A-J])\b")


def _extract_answer(text: Optional[str], kind: str) -> Optional[str]:
    if not text:
        return None
    if kind in ("mcq", "numeric"):
        match = _MC_ANSWER_RE.findall(text)
        # MCQ 优先取最后一个选项字母；数值/MATH 不抽取。
        if kind == "mcq" and match:
            return match[-1].upper()
    # MATH / GSM8K 继续交给仓库 evaluator，不在这里做字符串判分。
    return None


def _classify_error(exc: BaseException, status_code: Optional[int]) -> str:
    if status_code is not None:
        return "http_4xx" if 400 <= status_code < 500 else "http_5xx"
    msg = str(exc).lower()
    if "timeout" in msg:
        return "timeout"
    if any(k in msg for k in ("connection", "refused", "eof", "stream interrupted", "name or service")):
        return "connection"
    if "invalid json" in msg:
        return "invalid_response"
    return "request_error"


def _sample_memory_snapshot() -> Optional[Dict[str, Any]]:
    """轻量采集单条请求后的内存/swap 快照；失败返回 None。"""
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5.0)
        memsize = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True, timeout=5.0)
        swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                              text=True, timeout=5.0)
        if vm.returncode != 0 or memsize.returncode != 0:
            return None
        text = vm.stdout
        m = re.search(r"page size of\s+(\d+)", text)
        page_size = int(m.group(1)) if m else 16384

        def pages(label: str) -> int:
            mm = re.search(rf"{label}:\s+(\d+)", text)
            return int(mm.group(1)) if mm else 0

        free_gib = pages("Pages free") * page_size / (1024 ** 3)
        wired_gib = pages("Pages wired down") * page_size / (1024 ** 3)
        total_gib = int(memsize.stdout.strip()) / (1024 ** 3) if memsize.stdout.strip() else 0.0
        used_gib = max(0.0, total_gib - free_gib)

        swap_used = None
        mm = re.search(r"used:\s*([\d.]+)\s*([MG])i?B?", swap.stdout or "", re.IGNORECASE)
        if mm:
            value = float(mm.group(1))
            if mm.group(2).upper() == "G":
                value *= 1024.0
            swap_used = round(value, 1)

        return {
            "memory_used_gib": round(used_gib, 3),
            "memory_free_gib": round(free_gib, 3),
            "wired_gib": round(wired_gib, 3),
            "swap_used_mib": swap_used,
        }
    except Exception:
        return None


def _run_one_attempt(client: OpenAICompatClient, query: Dict[str, Any],
                     mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    messages = [{"role": "user", "content": query["prompt"]}]
    spec = DATASET_SPECS[query["dataset_id"]]
    identity = {
        "run_id": getattr(args, "run_id", None),
        "host": getattr(args, "node_id", None),
        "engine": args.engine,
        "engine_version": args.engine_version,
        "weight_format": getattr(args, "weight_format", None),
        "quantization": getattr(args, "quantization", None),
        "draft_model": getattr(args, "draft_model", None),
        "kv_format": getattr(args, "kv_format", None),
        "state_format": getattr(args, "state_format", None),
        "model_revision_or_digest": getattr(args, "model_revision_or_digest", None),
    }
    try:
        if mode == "stream":
            raw = client.chat_stream(messages=messages, max_tokens=args.max_output_tokens,
                                     temperature=args.temperature, top_p=args.top_p,
                                     seed=args.seed)
        else:
            raw = client.chat_nonstream(messages=messages, max_tokens=args.max_output_tokens,
                                        temperature=args.temperature, top_p=args.top_p,
                                        seed=args.seed)
        answer_text = raw.get("content")
        expected = query.get("expected_answer")
        extracted = _extract_answer(answer_text, spec["kind"])
        hit = None
        if expected and extracted is not None:
            hit = extracted == expected.upper()

        if hit is True:
            correct = 1
            fail_reason = ""
        elif raw.get("success") is False:
            correct = 0
            fail_reason = raw.get("error") or "request_failed"
        elif hit is False:
            correct = 0
            fail_reason = "wrong_answer"
        elif extracted is None:
            correct = 0
            fail_reason = "parse_error" if answer_text else "no_answer"
        else:
            correct = 0
            fail_reason = "unjudgeable"

        prompt_tokens = raw.get("prompt_tokens")
        completion_tokens = raw.get("completion_tokens")
        cache_hit_tokens = raw.get("cache_hit_tokens")
        if cache_hit_tokens is not None and prompt_tokens is not None:
            prefill_tokens = max(0, prompt_tokens - cache_hit_tokens)
        else:
            prefill_tokens = prompt_tokens

        prefill_s = raw.get("prefill_s")
        decode_s = raw.get("decode_s")
        prefill_tps = None
        if prefill_s and prefill_s > 0 and prefill_tokens:
            prefill_tps = prefill_tokens / prefill_s
        decode_tps = None
        if decode_s and decode_s > 0 and completion_tokens:
            decode_tps = completion_tokens / decode_s

        machine = None
        if getattr(args, "per_request_machine_metrics", False):
            machine = _sample_memory_snapshot()

        result = {
            "sample_id": query.get("sample_id"),
            "eval_name": query["dataset_id"],
            "model_name": args.model,
            "prompt": query["prompt"],
            "response": answer_text or None,
            "correct": correct,
            "predicted_answer": extracted if extracted is not None else (answer_text or None),
            "gold_answer": expected or None,
            "fail_reason": fail_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": raw.get("reasoning_tokens"),
            "prefill_tokens": prefill_tokens,
            "decode_tokens": completion_tokens,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "request_start_time": raw.get("request_start_time"),
            "first_token_time": raw.get("first_token_time"),
            "response_end_time": raw.get("response_end_time"),
            "server_wall_s": raw.get("server_wall_s"),
            "client_e2e_s": raw.get("e2e_s"),
            "cache_status": raw.get("cache_status"),
            "cache_hit_tokens": cache_hit_tokens,
            "error": raw.get("error") or "",
            "error_type": None,
            "http_status": None,
            "timeout_setting_s": args.timeout,
            "dataset_id": query["dataset_id"],
            "record_index": query["record_index"],
            "mode": mode,
            "expected_answer": expected or None,
            "answer_text": answer_text or None,
            "extracted_answer": extracted,
            "answer_hit": hit,
            "success": bool(raw.get("success")),
            "finish_reason": raw.get("finish_reason"),
            "ttft_s": raw.get("ttft_s"),
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "prefill_tok_per_s": raw.get("prefill_tok_per_s"),
            "decode_tok_per_s": raw.get("decode_tok_per_s"),
            "e2e_s": raw.get("e2e_s"),
            "timing_source": raw.get("timing_source"),
            "per_request_machine": machine,
        }
        result.update(identity)
        return result
    except BenchRequestError as exc:
        status_code = getattr(exc, "status_code", None)
        machine = None
        if getattr(args, "per_request_machine_metrics", False):
            machine = _sample_memory_snapshot()
        result = {
            "sample_id": query.get("sample_id"),
            "eval_name": query["dataset_id"],
            "model_name": args.model,
            "prompt": query["prompt"],
            "response": None,
            "correct": 0,
            "predicted_answer": None,
            "gold_answer": query.get("expected_answer") or None,
            "fail_reason": str(exc)[:500] or "request_failed",
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": -1,
            "prefill_tokens": None,
            "decode_tokens": None,
            "prefill_tps": None,
            "decode_tps": None,
            "request_start_time": None,
            "first_token_time": None,
            "response_end_time": None,
            "server_wall_s": None,
            "client_e2e_s": None,
            "cache_status": None,
            "cache_hit_tokens": None,
            "error": str(exc)[:500],
            "error_type": _classify_error(exc, status_code),
            "http_status": status_code,
            "timeout_setting_s": args.timeout,
            "dataset_id": query["dataset_id"],
            "record_index": query["record_index"],
            "mode": mode,
            "expected_answer": query.get("expected_answer") or None,
            "answer_text": None,
            "extracted_answer": None,
            "answer_hit": None,
            "success": False,
            "finish_reason": None,
            "ttft_s": None,
            "prefill_s": None,
            "decode_s": None,
            "prefill_tok_per_s": None,
            "decode_tok_per_s": None,
            "e2e_s": None,
            "timing_source": None,
            "per_request_machine": machine,
        }
        result.update(identity)
        return result



def run_one_mode(client: OpenAICompatClient, query: Dict[str, Any],
                 mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    """带退避重试的单次评测请求；记录 attempt_index / retry_count / backoff_s。"""
    max_attempts = max(1, int(getattr(args, "max_attempts", 1)))
    backoff_base = max(0.0, float(getattr(args, "backoff_base_s", 1.0)))
    slept = 0.0
    last: Optional[Dict[str, Any]] = None
    for attempt in range(max_attempts):
        result = _run_one_attempt(client, query, mode, args)
        result["attempt_index"] = attempt
        result["retry_count"] = attempt
        result["backoff_s"] = round(slept, 3)
        if result.get("success"):
            return result
        last = result
        if attempt < max_attempts - 1:
            slept = min(30.0, backoff_base * (2 ** attempt))
            time.sleep(slept)
        else:
            slept = 0.0
    if last is not None:
        last["retry_count"] = max_attempts - 1
        return last
    result = _run_one_attempt(client, query, mode, args)
    result["attempt_index"] = max_attempts - 1
    result["retry_count"] = max_attempts - 1
    result["backoff_s"] = 0.0
    return result



def _p95(values: List[float]) -> Optional[float]:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    idx = max(0, int(len(clean) * 0.95 + 0.5) - 2)
    return clean[idx]


def _mean(values: List[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def summarize_requests(requests: List[Dict[str, Any]], wall_s: float, concurrency: int) -> Dict[str, Any]:
    ok = [r for r in requests if r.get("success")]
    stream = [r for r in requests if r.get("mode") == "stream"]
    nonstream = [r for r in requests if r.get("mode") == "nonstream"]
    ss = [r for r in ok if r.get("mode") == "stream"]
    prompt_tokens = [r.get("prompt_tokens") for r in ok if r.get("prompt_tokens") is not None]
    completion_tokens = [r.get("completion_tokens") for r in ok if r.get("completion_tokens") is not None]
    ttfts = [r.get("ttft_s") for r in ss if r.get("ttft_s") is not None]
    decode_tps = [r.get("decode_tok_per_s") for r in ok if r.get("decode_tok_per_s") is not None]
    prefill_tps = [r.get("prefill_tok_per_s") for r in ok if r.get("prefill_tok_per_s") is not None]
    return {
        "concurrency": concurrency,
        "wall_s": round(wall_s, 3),
        "total_requests": len(requests),
        "success_requests": len(ok),
        "error_requests": len(requests) - len(ok),
        "success_rate": round(len(ok) / len(requests), 4) if requests else None,
        "requests_per_s": round(len(ok) / wall_s, 3) if wall_s > 0 else None,
        "total_prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "total_completion_tokens": sum(completion_tokens) if completion_tokens else None,
        "tokens_per_s": round((sum(prompt_tokens) + sum(completion_tokens)) / wall_s, 2)
                        if wall_s > 0 and (prompt_tokens or completion_tokens) else None,
        "ttft_s_p95": round(_p95(ttfts), 4) if ttfts else None,
        "ttft_s_mean": round(_mean(ttfts), 4) if ttfts else None,
        "decode_tok_per_s_mean": round(_mean(decode_tps), 2) if decode_tps else None,
        "prefill_tok_per_s_mean": round(_mean(prefill_tps), 2) if prefill_tps else None,
        "stream_requests": len(stream),
        "nonstream_requests": len(nonstream),
    }


def run_stage(concurrency: int, queries: List[Dict[str, Any]], client: OpenAICompatClient,
              args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Optional[float]]:
    modes: List[str] = []
    if args.mode in ("nonstream", "both"):
        modes.append("nonstream")
    if args.mode in ("stream", "both"):
        modes.append("stream")

    # 预热：当前并发对首批 query 做 1 次请求。
    if queries and args.warmup > 0:
        warm_q = queries[: min(concurrency, len(queries))]
        warm_tasks = [(query, mode) for _ in range(args.warmup) for query in warm_q for mode in modes]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(ex.map(lambda item: run_one_mode(client, item[0], item[1], args), warm_tasks))

    tasks = [(query, mode) for _ in range(args.repeat)
             for query in queries for mode in modes]

    monitor = SystemMonitor(pid=getattr(args, "server_pid", None),
                            disk=getattr(args, "monitor_disk", None))
    monitor.start()
    started = time.perf_counter()
    requests: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_one_mode, client, q, m, args) for q, m in tasks]
        for fut in concurrent.futures.as_completed(futures):
            try:
                requests.append(fut.result())
            except Exception as exc:  # 防御：收集意外异常
                requests.append({
                    "dataset_id": None, "record_index": None, "mode": None,
                    "expected_answer": None, "answer_text": None,
                    "extracted_answer": None, "answer_hit": None,
                    "success": False, "error": f"future_error: {exc}", "finish_reason": None,
                    "prompt_tokens": None, "completion_tokens": None, "ttft_s": None,
                    "prefill_s": None, "decode_s": None, "prefill_tok_per_s": None,
                    "decode_tok_per_s": None, "e2e_s": None, "timing_source": None,
                })
    wall_s = time.perf_counter() - started
    memory, ssd_read = monitor.stop()
    summary = summarize_requests(requests, wall_s, concurrency)
    summary["memory"] = memory
    summary["ssd_read_mib_s"] = ssd_read
    if getattr(args, "flops_per_token", None) and summary.get("tokens_per_s"):
        summary["estimated_tflops"] = round(
            summary["tokens_per_s"] * args.flops_per_token / 1e12, 6)
        summary["estimated_pflops"] = round(
            summary["tokens_per_s"] * args.flops_per_token / 1e15, 6)
    return summary, requests, ({},)


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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench_multi_query",
        description="能力评测集并发阶梯多 query 性能测量")
    parser.add_argument("--model", required=True, help="API 模型 ID")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--run-id", default=None, help="评测运行 ID；缺省自动生成")
    parser.add_argument("--node-id", default=None, help="节点 ID；缺省使用本机 hostname")
    parser.add_argument("--engine", default="splash")
    parser.add_argument("--engine-version", default="1.5.0")
    parser.add_argument("--weight-format", default=None, help="gguf / mlx")
    parser.add_argument("--quantization", default=None, help="模型量化单位，写入 manifest 与每条请求")
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--kv-format", default="int8")
    parser.add_argument("--state-format", default="bf16")
    parser.add_argument("--model-revision-or-digest", default=None)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                        help="逗号分隔 dataset_id，默认六类能力集")
    parser.add_argument("--limit-per-dataset", type=int, default=20,
                        help="每数据集采样条数；0 表示全量（默认 20）")
    parser.add_argument("--concurrency", default="1,2,4,8,16",
                        help="逗号分隔并发阶梯")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="每请求最多尝试次数（首次 + 重试）")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--mode", choices=("stream", "nonstream", "both"), default="both")
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--backoff-base-s", type=float, default=1.0,
                        help="失败重试退避基数（指数退避，上限 30s）")
    parser.add_argument("--per-request-machine-metrics", action="store_true",
                        help="每条请求后额外采样内存/swap 快照")
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--monitor-disk", default=None)
    parser.add_argument("--flops-per-token", type=float, default=None,
                        help="每个 token 的有效浮点运算数，用于估算 TFLOPS/PFLOPS")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="只加载能力集并打印统计，不请求服务")
    args = parser.parse_args(argv)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    concurrency = [int(x) for x in args.concurrency.split(",") if x.strip()]
    queries = load_queries(datasets, args.limit_per_dataset)

    tag = _safe_name(args.model)
    out_dir = args.out or os.path.join("results", "mac_benchmark", "multi_query", tag)
    if args.dry_run:
        print(f"[dry-run] model={args.model} port={args.port}")
        print(f"[dry-run] datasets={datasets} limit_per_dataset={args.limit_per_dataset}")
        print(f"[dry-run] queries={len(queries)} concurrency={concurrency} repeat={args.repeat}")
        per = {}
        for q in queries:
            per[q["dataset_id"]] = per.get(q["dataset_id"], 0) + 1
        print("[dry-run] per_dataset=" + json.dumps(per, ensure_ascii=False))
        print("[dry-run] first_prompt=" + queries[0]["prompt"][:200].replace("\n", " "))
        return 0

    client = OpenAICompatClient(host=args.host, port=args.port, model=args.model,
                                timeout_s=args.timeout)

    hardware = detect_hardware()
    if not args.run_id:
        args.run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + tag
    args.node_id = args.node_id or hardware.get("host") or args.host
    args.hardware = hardware

    manifest = {
        "tool": "bench_multi_query",
        "run_id": args.run_id,
        "node_id": args.node_id,
        "engine": args.engine,
        "engine_version": args.engine_version,
        "model": args.model,
        "weight_format": args.weight_format,
        "quantization": args.quantization,
        "draft_model": args.draft_model,
        "kv_format": args.kv_format,
        "state_format": args.state_format,
        "model_revision_or_digest": args.model_revision_or_digest,
        "host": args.host,
        "port": args.port,
        "datasets": datasets,
        "limit_per_dataset": args.limit_per_dataset,
        "concurrency": concurrency,
        "repeat": args.repeat,
        "max_attempts": args.max_attempts,
        "warmup": args.warmup,
        "mode": args.mode,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "flops_per_token": args.flops_per_token,
        "max_attempts": args.max_attempts,
        "backoff_base_s": args.backoff_base_s,
        "per_request_machine_metrics": args.per_request_machine_metrics,
        "query_count": len(queries),
        "hardware": hardware,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)
    print(f"[bench_multi_query] model={args.model} queries={len(queries)} "
          f"stages={concurrency} repeat={args.repeat} out={out_dir}")

    for c in concurrency:
        print(f"[bench_multi_query] stage concurrency={c} start")
        summary, requests, _ = run_stage(c, queries, client, args)
        write_json(os.path.join(out_dir, f"stage_{c}.json"), summary)
        write_jsonl(os.path.join(out_dir, f"stage_{c}_requests.jsonl"), requests)
        thr = summary.get("requests_per_s")
        tps = summary.get("tokens_per_s")
        err = summary.get("error_requests")
        print(f"[bench_multi_query] stage c={c} ok={summary.get('success_requests')}/{summary.get('total_requests')} "
              f"req/s={thr} tok/s={tps} errors={err}")

    print(f"[bench_multi_query] done, {len(concurrency)} stage(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
