# -*- coding: utf-8 -*-
"""对比两种 GPU 负载百分比画法。"""
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
    p.add_argument("--out-effective-png", required=True)
    p.add_argument("--out-power-png", required=True)
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

    xs, eff, pw = [], [], []
    for r in metrics:
        t = r.get("wall_time")
        if t is None or not (t0 <= float(t) <= t1):
            continue
        g = r.get("gpu") or {}
        u = g.get("gpu_usage_percent")
        p_mw = g.get("gpu_power_mw")
        if u is None or p_mw is None:
            continue
        xs.append((float(t) - t0) / 60.0)
        ratio = p_mw / (args.power_ceiling_w * 1000.0)
        eff.append(u * ratio)          # 0..100
        pw.append(ratio * 100.0)       # 0..100

    wins = stage_windows(workload)
    ordered = sorted(wins)
    boundaries = []
    labels = []
    for i, c in enumerate(ordered):
        start, end = wins[c]
        labels.append(((((start - t0) / 60.0) + ((end - t0) / 60.0)) / 2.0, f"c={c}"))
        if i > 0:
            boundaries.append((start - t0) / 60.0)

    def draw(values, ylabel, title, out_png):
        fig, ax = plt.subplots(figsize=(13, 5.0))
        ax.plot(xs, values, linewidth=0.9, alpha=0.9, color="tab:green")
        for b in boundaries:
            ax.axvline(b, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
        for mid, label in labels:
            ax.text(mid, max(values) * 1.03, label, ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("relative time (min)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    draw(eff, "GPU effective load % (residency × power ratio)",
         f"{args.run_label} | GPU effective load", args.out_effective_png)
    draw(pw, "GPU power utilization % (power / ceiling)",
         f"{args.run_label} | GPU power utilization", args.out_power_png)
    print(f"[done] effective={args.out_effective_png}")
    print(f"[done] power={args.out_power_png}")


if __name__ == "__main__":
    main()
