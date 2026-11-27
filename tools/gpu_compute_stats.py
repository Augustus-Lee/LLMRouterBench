# -*- coding: utf-8 -*-
"""从 stage_*.json 统计 GPU/工作负载的有效算力。

有效算力 = token 吞吐 × 每 token FLOPs。

默认 FLOPs/token 按 dense 模型近似：
    FLOPs_per_token = 2 × active_params

可显式覆盖 --flops-per-token，以适配 MoE 或其它口径。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt(v: Optional[float], nd: int = 3) -> str:
    return f"{v:.{nd}f}" if v is not None else "-"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="统计 GPU 有效算力")
    p.add_argument("--stages-dir", required=True, help="包含 stage_*.json 的目录")
    p.add_argument("--active-params-b", type=float, default=27.0,
                   help="模型激活参数量（B），仅用于默认 FLOPs/token")
    p.add_argument("--flops-per-token", type=float, default=None,
                   help="显式指定每 token FLOPs")
    p.add_argument("--peak-tflops", type=float, default=None,
                   help="GPU 峰值算力 TFLOPS；缺省时 utilization 为 null")
    p.add_argument("--out-md", required=True, help="输出 Markdown 路径")
    args = p.parse_args(argv)

    flops_per_token = args.flops_per_token
    if flops_per_token is None:
        flops_per_token = 2.0 * args.active_params_b * 1e9

    stage_files = sorted(glob.glob(os.path.join(args.stages_dir, "stage_*.json")))
    if not stage_files:
        print(f"未找到 stage JSON: {args.stages_dir}")
        return 2

    lines = [
        "# GPU 有效算力统计",
        "",
        f"> flops_per_token = {flops_per_token:.6e}",
        f"> peak_tflops = {args.peak_tflops if args.peak_tflops is not None else 'null'}",
        "",
        "| 并发 | wall_tflops | decode_mean_tflops | prefill_mean_tflops | compute_utilization_pct(value/tflops) |",
        "|---|---:|---:|---:|---:|",
    ]

    overall_wall_s = 0.0
    overall_total_tokens = 0.0

    for path in stage_files:
        d = load_json(path)
        c = d.get("concurrency")
        wall_s = num(d.get("wall_s")) or 0.0
        total_tokens = 0.0
        pt = d.get("total_prompt_tokens")
        ct = d.get("total_completion_tokens")
        if num(pt) is not None:
            total_tokens += num(pt)
        if num(ct) is not None:
            total_tokens += num(ct)

        tokens_per_s = num(d.get("tokens_per_s"))
        decode_mean = num(d.get("decode_tok_per_s_mean"))
        prefill_mean = num(d.get("prefill_tok_per_s_mean"))

        wall_tflops = (tokens_per_s * flops_per_token / 1e12) if tokens_per_s is not None else None
        decode_tflops = (decode_mean * flops_per_token / 1e12) if decode_mean is not None else None
        prefill_tflops = (prefill_mean * flops_per_token / 1e12) if prefill_mean is not None else None

        effective = decode_tflops if decode_tflops is not None else wall_tflops
        util = None
        if effective is not None and args.peak_tflops and args.peak_tflops > 0:
            util = effective / args.peak_tflops * 100.0

        lines.append(
            f"| {c} | {fmt(wall_tflops)} | {fmt(decode_tflops)} "
            f"| {fmt(prefill_tflops)} | {fmt(util, 2)} |"
        )

        overall_wall_s += wall_s
        overall_total_tokens += total_tokens

    if overall_wall_s > 0:
        overall_tps = overall_total_tokens / overall_wall_s
        overall_tflops = overall_tps * flops_per_token / 1e12
        overall_util = None
        if args.peak_tflops and args.peak_tflops > 0:
            overall_util = overall_tflops / args.peak_tflops * 100.0
        lines.extend([
            "",
            "## 总体按 wall time 汇总",
            "",
            f"- total_wall_s = {overall_wall_s:.2f}",
            f"- total_tokens = {overall_total_tokens:.0f}",
            f"- tokens_per_s = {overall_tps:.3f}",
            f"- effective_tflops = {overall_tflops:.4f}",
            f"- compute_utilization_pct = {fmt(overall_util, 2)}",
            "",
        ])

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"[gpu_compute_stats] report={args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
