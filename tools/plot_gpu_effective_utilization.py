# -*- coding: utf-8 -*-
"""绘制 GPU effective utilization 随时间变化图。"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, help="gpu_metrics_stream 输出的 JSONL")
    p.add_argument("--out-png", required=True)
    p.add_argument("--run-label", default="")
    args = p.parse_args()

    rows = load_jsonl(args.metrics)
    xs, ys = [], []
    base_wall = None
    for r in rows:
        gpu = r.get("gpu") or {}
        t = r.get("wall_time")
        v = gpu.get("gpu_effective_utilization_percent")
        if t is None or v is None:
            continue
        if base_wall is None:
            base_wall = float(t)
        xs.append((float(t) - base_wall) / 60.0)
        ys.append(float(v))

    if not xs:
        print("no usable gpu effective utilization samples")
        return 2

    median = statistics.median(ys)
    mean = statistics.fmean(ys)
    p95 = sorted(ys)[int(len(ys) * 0.95)]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(xs, ys, linewidth=0.9, alpha=0.9, color="tab:blue")
    ax.axhline(median, color="tab:orange", linestyle="--", linewidth=1.0,
               label=f"median={median:.1f}%")
    ax.axhline(p95, color="tab:red", linestyle=":", linewidth=1.0,
               label=f"P95={p95:.1f}%")
    ax.set_xlabel("relative time (min)")
    ax.set_ylabel("GPU effective utilization (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{args.run_label} | GPU effective utilization".strip())
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"[done] samples={len(xs)} mean={mean:.2f}% median={median:.2f}% "
          f"P95={p95:.2f}% png={args.out_png}")


if __name__ == "__main__":
    main()
