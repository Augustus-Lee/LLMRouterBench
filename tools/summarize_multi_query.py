# -*- coding: utf-8 -*-
"""并发多 Query 评测结果汇总与成图。

输入：`tools/bench_multi_query.py` 输出的目录，例如
``results/mac_benchmark/multi_query/qwen38``，其中包含：

- ``manifest.json``
- ``stage_<concurrency>.json``
- ``stage_<concurrency>_requests.jsonl``

输出：

- ``summary.md``：每个并发档位的吞吐、TTFT、decode、错误、内存汇总
- ``charts/``：并发-吞吐/TTFT/decode/内存曲线（matplotlib 可选）

用法：
    python tools/summarize_multi_query.py \
        --result-dir results/mac_benchmark/multi_query/qwen38 \
        --out results/mac_benchmark/multi_query/qwen38/summary \
        --chart

也可扫描多个模型目录：
    python tools/summarize_multi_query.py \
        --result-root results/mac_benchmark/multi_query \
        --out results/mac_benchmark/multi_query_summary \
        --chart
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _stage_name(path: str) -> int:
    base = os.path.basename(path)
    if base.endswith(".json"):
        base = base[:-5]
    return int(base.split("_", 1)[1])


def load_result_dir(path: str) -> Dict[str, Any]:
    manifest = _load_json(os.path.join(path, "manifest.json")) or {}
    stages: List[Dict[str, Any]] = []
    for stage_path in sorted(glob.glob(os.path.join(path, "stage_*.json")),
                             key=_stage_name):
        if "_requests.json" in stage_path:
            continue
        stage = _load_json(stage_path) or {}
        req_path = stage_path.replace(".json", "_requests.jsonl")
        requests = _load_jsonl(req_path) if os.path.exists(req_path) else []
        stage["requests"] = requests
        stage["concurrency"] = stage.get("concurrency") or _stage_name(stage_path)
        stages.append(stage)
    return {"manifest": manifest, "stages": stages, "path": path}


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _mean_per_request(requests: List[Dict[str, Any]], field_path: List[str]) -> Optional[float]:
    vals = []
    for r in requests:
        cur: Any = r
        for key in field_path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(key)
        if cur is None:
            continue
        try:
            vals.append(float(cur))
        except (TypeError, ValueError):
            continue
    return sum(vals) / len(vals) if vals else None


def _correct_counts(requests: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    correct = wrong = unjudged = 0
    for r in requests:
        if not r.get("success"):
            unjudged += 1
        elif r.get("correct") == 1:
            correct += 1
        elif r.get("correct") == 0:
            wrong += 1
        else:
            unjudged += 1
    return correct, wrong, unjudged


def render_markdown(meta: Dict[str, Any]) -> str:
    manifest = meta["manifest"]
    model = manifest.get("model") or os.path.basename(meta["path"])
    engine = manifest.get("engine") or "-"
    quantization = manifest.get("quantization") or "-"
    lines = [
        f"# 并发多 Query 评测汇总 — {model}",
        "",
        f"- engine: `{engine}`",
        f"- quantization: `{quantization}`",
        f"- run_id: `{manifest.get('run_id') or '-'}`",
        f"- source: `{meta['path']}`",
        "",
        "| 并发 | 请求数 | 成功率 | req/s | tok/s | TTFT mean | TTFT p95 | Decode tok/s | Prefill tok/s | 错误数 | 内存 used GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in meta["stages"]:
        reqs = stage.get("requests", [])
        mem_used = stage.get("per_request_memory_used_gib")
        if mem_used is None:
            mem_used = _mean_per_request(reqs, ["per_request_machine", "memory_used_gib"])
        lines.append(
            f"| {stage.get('concurrency')} "
            f"| {stage.get('total_requests')} "
            f"| {_fmt(stage.get('success_rate'))} "
            f"| {_fmt(stage.get('requests_per_s'))} "
            f"| {_fmt(stage.get('tokens_per_s'))} "
            f"| {_fmt(stage.get('ttft_s_mean'), 3)} "
            f"| {_fmt(stage.get('ttft_s_p95'), 3)} "
            f"| {_fmt(stage.get('decode_tok_per_s_mean'))} "
            f"| {_fmt(stage.get('prefill_tok_per_s_mean'))} "
            f"| {stage.get('error_requests') or 0} "
            f"| {_fmt(mem_used)} |"
        )
    lines.append("")
    lines.append("## 能力判分摘要")
    lines.append("")
    lines.append("| 并发 | correct | wrong | unjudged |")
    lines.append("|---:|---:|---:|---:|")
    for stage in meta["stages"]:
        c, w, u = _correct_counts(stage.get("requests", []))
        lines.append(f"| {stage.get('concurrency')} | {c} | {w} | {u} |")
    return "\n".join(lines) + "\n"


def _series(metas: List[Dict[str, Any]], field: str):
    series: List[Tuple[str, List[int], List[float]]] = []
    for meta in metas:
        label = meta["manifest"].get("model") or os.path.basename(meta["path"])
        xs, ys = [], []
        for stage in meta["stages"]:
            value = stage.get(field)
            if value is None:
                continue
            xs.append(int(stage.get("concurrency")))
            ys.append(float(value))
        if xs:
            series.append((label, xs, ys))
    return series


def make_charts(metas: List[Dict[str, Any]], out_dir: str) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[summarize_multi_query] matplotlib 不可用，跳过图表")
        return []
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    saved: List[str] = []

    def plot(field, ylabel, fname, title):
        series = _series(metas, field)
        if not series:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, xs, ys in series:
            ax.plot(xs, ys, marker="o", label=label[:50])
        ax.set_xlabel("concurrency")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(charts_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    plot("requests_per_s", "requests / s", "requests_per_s.png",
         "Request throughput vs concurrency")
    plot("tokens_per_s", "tokens / s", "tokens_per_s.png",
         "Token throughput vs concurrency")
    plot("ttft_s_p95", "TTFT p95 (s)", "ttft_p95.png",
         "TTFT p95 vs concurrency")
    plot("decode_tok_per_s_mean", "Decode tok/s (mean)", "decode_tps.png",
         "Decode throughput vs concurrency")
    plot("prefill_tok_per_s_mean", "Prefill tok/s (mean)", "prefill_tps.png",
         "Prefill throughput vs concurrency")
    return saved


def _discover_dirs(root: str) -> List[str]:
    dirs: List[str] = []
    for path in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(path) and glob.glob(os.path.join(path, "stage_*.json")):
            dirs.append(path)
    return dirs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="summarize_multi_query",
        description="汇总 bench_multi_query 输出并生成图表")
    parser.add_argument("--result-dir", default=None, help="单个结果目录")
    parser.add_argument("--result-root", default=None, help="扫描多个模型结果目录的根目录")
    parser.add_argument("--out", default=None)
    parser.add_argument("--chart", action="store_true")
    args = parser.parse_args(argv)

    if args.result_dir:
        dirs = [args.result_dir]
    elif args.result_root:
        dirs = _discover_dirs(args.result_root)
        if not dirs:
            print(f"[summarize_multi_query] no result dirs under {args.result_root}")
            return 1
    else:
        print("[summarize_multi_query] need --result-dir or --result-root")
        return 1

    metas = [load_result_dir(d) for d in dirs if d]
    out_dir = args.out or (args.result_dir + "/summary" if args.result_dir else args.result_root + "/summary")
    os.makedirs(out_dir, exist_ok=True)

    for meta in metas:
        model = meta["manifest"].get("model") or os.path.basename(meta["path"])
        md_path = os.path.join(out_dir, f"{os.path.basename(meta['path'])}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(render_markdown(meta))
        print(f"[summarize_multi_query] wrote {md_path}")

    charts = make_charts(metas, out_dir) if args.chart else []
    if charts:
        print(f"[summarize_multi_query] charts={len(charts)} -> {out_dir}/charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
