# -*- coding: utf-8 -*-
"""绘制 GPU / CPU 利用率随并发阶段变化图，并统计 GPU full_load 窗口。"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

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


def stage_windows(workload: List[Dict[str, Any]]) -> Dict[int, Tuple[float, float]]:
    by: Dict[int, List[float]] = {}
    for e in workload:
        c = e.get("concurrency")
        t = e.get("wall_time")
        if c is None or t is None:
            continue
        by.setdefault(int(c), []).append(float(t))
    out = {}
    for c, times in by.items():
        out[c] = (min(times) - 5.0, max(times) + 5.0)
    return out


def main() -> int:
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
        print("metrics empty")
        return 2

    events = [e for e in workload if e.get("wall_time") is not None]
    if events:
        t0 = min(float(e["wall_time"]) for e in events) - 5.0
        t1 = max(float(e["wall_time"]) for e in events) + 5.0
    else:
        ts = [float(r.get("wall_time")) for r in metrics if r.get("wall_time") is not None]
        t0, t1 = min(ts), max(ts)

    xs, gpu, cpu, full = [], [], [], []
    wall_ts = []
    for r in metrics:
        t = r.get("wall_time")
        if t is None or not (t0 <= float(t) <= t1):
            continue
        g = r.get("gpu") or {}
        c = r.get("cpu") or {}
        u = g.get("gpu_usage_percent")
        cu = c.get("used_percent")
        pwr = g.get("gpu_power_mw")
        if u is None and cu is None:
            continue
        wall_ts.append(float(t))
        xs.append((float(t) - t0) / 60.0)
        gpu.append(u)
        cpu.append(cu)
        ok = (u is not None and u >= 95.0 and pwr is not None and pwr >= 0.90 * args.power_ceiling_w * 1000.0)
        full.append(1.0 if ok else 0.0)

    # 60s rolling window percentage for full_load
    window_s = 60.0
    full_pct = []
    n = len(full)
    for i in range(n):
        left = i
        while left >= 0 and wall_ts[i] - wall_ts[left] <= window_s:
            left -= 1
        left += 1
        full_pct.append(100.0 * sum(full[left:i+1]) / (i - left + 1))

    # stage boundaries and labels
    wins = stage_windows(workload)
    ordered = sorted(wins)
    boundaries = []
    labels = []
    for i, c in enumerate(ordered):
        start, end = wins[c]
        mid = ((start - t0) / 60.0 + (end - t0) / 60.0) / 2.0
        labels.append((mid, f"concurrency={c}"))
        if i > 0:
            boundaries.append((start - t0) / 60.0)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(xs, gpu, linewidth=0.9, alpha=0.85, color="tab:blue", label="GPU active residency (%)")
    ax.plot(xs, cpu, linewidth=0.8, alpha=0.55, color="tab:orange", label="CPU used (%)")
    ax.plot(xs, full_pct, linewidth=1.1, alpha=0.9, color="tab:green",
            label="GPU full_load % (60s rolling)")

    y = [0, 100]
    xarr = xs
    full_bool = [v > 50.0 for v in full_pct]
    ax.fill_between(xarr, 0, 100, where=full_bool, color="green", alpha=0.05,
                    label=f"full_load > 50%")

    for b in boundaries:
        ax.axvline(b, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)

    ylim = (0, 100)
    for mid, label in labels:
        ax.text(mid, 102, label, ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("relative time (min)")
    ax.set_ylabel("utilization (%)")
    title = f"{args.run_label} GPU vs CPU utilization"
    ax.set_title(title)
    ax.set_ylim(0, 108)
    ax.set_xlim(0, max(xs) if xs else 0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)

    # summary
    print(f"[done] rows={len(xs)} stages={ordered}")
    print(f"[done] png={args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
