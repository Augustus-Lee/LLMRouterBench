# -*- coding: utf-8 -*-
"""绘制 CPU 使用率 + GPU active residency + GPU power utilization。"""
from __future__ import annotations

import argparse
import json
import os
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


def stage_windows(workload):
    by: Dict[int, List[float]] = {}
    for e in workload:
        c = e.get("concurrency")
        t = e.get("wall_time")
        if c is None or t is None:
            continue
        by.setdefault(int(c), []).append(float(t))
    return {c: (min(v) - 5.0, max(v) + 5.0) for c, v in by.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--out-png", required=True)
    p.add_argument("--power-ceiling-w", type=float, default=37.4)
    p.add_argument("--run-label", default="")
    args = p.parse_args()

    metrics = load_jsonl(args.metrics)
    workload = load_jsonl(args.workload)
    if not metrics:
        return 2

    events = [e for e in workload if e.get("wall_time") is not None]
    t0 = min(float(e["wall_time"]) for e in events) - 5.0
    t1 = max(float(e["wall_time"]) for e in events) + 5.0

    xs, gpu, cpu, power_util = [], [], [], []
    for r in metrics:
        t = r.get("wall_time")
        if t is None or not (t0 <= float(t) <= t1):
            continue
        g = r.get("gpu") or {}
        c = r.get("cpu") or {}
        u = g.get("gpu_usage_percent")
        pwr = g.get("gpu_power_mw")
        cu = c.get("used_percent")
        if u is None and pwr is None and cu is None:
            continue
        xs.append((float(t) - t0) / 60.0)
        gpu.append(u)
        cpu.append(cu)
        power_util.append((pwr / (args.power_ceiling_w * 1000.0) * 100.0) if pwr is not None else None)

    wins = stage_windows(workload)
    ordered = sorted(wins)
    boundaries, labels = [], []
    for i, c in enumerate(ordered):
        start, end = wins[c]
        labels.append(((((start - t0) / 60.0) + ((end - t0) / 60.0)) / 2.0, f"concurrency={c}"))
        if i > 0:
            boundaries.append((start - t0) / 60.0)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(xs, gpu, linewidth=0.9, alpha=0.85, color="tab:blue", label="GPU active residency (%)")
    ax.plot(xs, cpu, linewidth=0.8, alpha=0.55, color="tab:orange", label="CPU used (%)")
    ax.plot(xs, power_util, linewidth=1.1, alpha=0.9, color="tab:green",
            label=f"GPU power utilization % (power/{args.power_ceiling_w:.1f}W)")

    for b in boundaries:
        ax.axvline(b, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
    for mid, label in labels:
        ax.text(mid, 102, label, ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("relative time (min)")
    ax.set_ylabel("utilization (%)")
    ax.set_ylim(0, 108)
    ax.set_title(f"{args.run_label} | CPU / GPU residency / GPU power utilization")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"[done] png={args.out_png}")


if __name__ == "__main__":
    main()
