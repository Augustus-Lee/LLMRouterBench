# -*- coding: utf-8 -*-
"""从 monitor_machine 输出统计 GPU 是否真正“算力满载”。

口径：
  - busy：GPU active residency 接近 100%
  - full load：busy 且 GPU 功耗达到本机参考上限的 90% 以上

参考上限默认取“本轮活跃窗口内 GPU 功耗的 P95”，仅作为本机本次负载的
观察上限；若有官方/可信 TDP，请通过 --power-ceiling-w 传入。

输出：
  <out_dir>/gpu_utilization_report.md
  <out_dir>/gpu_utilization_overlay.png
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def q(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(len(s) * p)))
    return s[idx]


def summarize_series(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": round(statistics.fmean(s), 2),
        "median": round(statistics.median(s), 2),
        "p95": round(q(s, 0.95), 2),
        "max": round(s[-1], 2),
    }


def fmt(v: Optional[float], nd: int = 2) -> str:
    return f"{v:.{nd}f}" if v is not None else "-"


def parse_stage_windows(workload: List[Dict[str, Any]]) -> Dict[int, Tuple[float, float]]:
    """按 workload 事件的 concurrency 字段推断每个 stage 的首尾 wall_time。"""
    by: Dict[int, List[float]] = {}
    for e in workload:
        c = e.get("concurrency")
        t = e.get("wall_time")
        if c is None or t is None:
            continue
        try:
            by.setdefault(int(c), []).append(float(t))
        except (TypeError, ValueError):
            continue
    windows: Dict[int, Tuple[float, float]] = {}
    for c, times in by.items():
        windows[c] = (min(times) - 5.0, max(times) + 5.0)
    return windows


def filter_rows(metrics: List[Dict[str, Any]], t0: float, t1: float) -> List[Dict[str, Any]]:
    out = []
    for r in metrics:
        t = r.get("wall_time")
        if t is None:
            continue
        if t0 <= float(t) <= t1:
            out.append(r)
    return out


def make_chart(metrics: List[Dict[str, Any]], out_png: str) -> None:
    if not HAS_MPL:
        return
    xs = []
    usage = []
    power = []
    freq = []
    base = None
    for r in metrics:
        g = r.get("gpu") or {}
        t = r.get("wall_time")
        u = g.get("gpu_usage_percent")
        p = g.get("gpu_power_mw")
        f = g.get("gpu_freq_mhz")
        if t is None:
            continue
        if base is None:
            base = float(t)
        xs.append(float(t) - base)
        usage.append(u)
        power.append(p)
        freq.append(f)
    if not xs:
        return
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(xs, usage, linewidth=0.9, alpha=0.85, label="GPU active residency (%)")
    ax1.plot(xs, freq, linewidth=0.8, alpha=0.55, label="GPU freq (MHz)")
    ax1.set_xlabel("relative seconds")
    ax1.set_ylabel("usage % / freq MHz", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(xs, power, linewidth=0.9, alpha=0.75, color="tab:red", label="GPU power (mW)")
    ax2.set_ylabel("GPU power (mW)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title("GPU utilization / power / frequency overlay")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_report(run_label: str, metrics_path: str, workload_path: str,
                  power_ceiling_mw: float, overall: Dict[str, Any],
                  stages: List[Dict[str, Any]], out_png: str,
                  power_ceiling_source: str) -> str:
    lines: List[str] = []
    lines.append("# GPU 利用率分析报告")
    lines.append("")
    lines.append(f"> run: `{run_label}`")
    lines.append("")
    lines.append("## 输入")
    lines.append("")
    lines.append(f"- metrics: `{metrics_path}`")
    lines.append(f"- workload: `{workload_path}`")
    lines.append(f"- power ceiling: {power_ceiling_mw:.1f} mW（{power_ceiling_source}）")
    lines.append("")
    lines.append("## 方法")
    lines.append("")
    lines.append("- `busy_fraction = count(gpu_usage_percent >= 95) / count(all)`")
    lines.append("- `full_load_fraction = count(gpu_usage_percent >= 95 AND gpu_power_mw >= 0.90 * power_ceiling) / count(all)`")
    lines.append("- 若未显式提供官方 TDP，`power_ceiling` 取本轮活跃窗口 GPU 功耗 P95，仅代表本轮观察上限。")
    lines.append("")
    lines.append("## 总体活跃窗口")
    lines.append("")
    lines.append("| 指标 | n | mean | median | p95 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key, label in [("gpu_usage_percent", "GPU active residency (%)"),
                       ("gpu_power_mw", "GPU power (mW)"),
                       ("gpu_freq_mhz", "GPU frequency (MHz)")]:
        s = overall[key]
        lines.append(f"| {label} | {s['n']} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['p95'])} | {fmt(s['max'])} |")
    lines.append("")
    lines.append(f"- busy_fraction: **{fmt(overall['busy_fraction'] * 100, 2)}%**")
    lines.append(f"- full_load_fraction: **{fmt(overall['full_load_fraction'] * 100, 2)}%**")
    lines.append("")
    lines.append("## 分阶段")
    lines.append("")
    if stages:
        lines.append("| 并发 | 样本数 | usage median | usage p95 | power median | power p95 | freq median | busy_fraction | full_load_fraction |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for st in stages:
            lines.append(
                f"| {st['concurrency']} | {st['n']} "
                f"| {fmt(st['gpu_usage_percent']['median'])}% "
                f"| {fmt(st['gpu_usage_percent']['p95'])}% "
                f"| {fmt(st['gpu_power_mw']['median'])} mW "
                f"| {fmt(st['gpu_power_mw']['p95'])} mW "
                f"| {fmt(st['gpu_freq_mhz']['median'])} MHz "
                f"| {fmt(st['busy_fraction'] * 100, 2)}% "
                f"| {fmt(st['full_load_fraction'] * 100, 2)}% |")
    else:
        lines.append("无可用 stage workload 数据。")
    lines.append("")
    if out_png and os.path.exists(out_png):
        lines.append(f"## 曲线图")
        lines.append("")
        lines.append(f"![GPU overlay]({os.path.basename(out_png)})")
    lines.append("")
    lines.append("## 解读")
    lines.append("")
    lines.append("- `GPU active residency` 接近 100% 只说明 GPU 一直在工作。")
    lines.append("- 必须结合 power/freq：如果 usage 高但 power 明显低于上限，通常代表访存/带宽瓶颈或低 ALU 占用，不能宣称“算力满载”。")
    lines.append("- 如果 usage 与 power 同时接近上限，才是接近计算饱和。")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="统计 GPU active residency / power / frequency")
    p.add_argument("--metrics", required=True, help="monitor_machine 的 metrics.jsonl")
    p.add_argument("--workload", required=True, help="workload_counters.jsonl")
    p.add_argument("--out-dir", required=True, help="输出目录")
    p.add_argument("--run-label", default="run")
    p.add_argument("--power-ceiling-w", type=float, default=None,
                   help="GPU 参考功耗上限（W）。缺省时使用活跃窗口 P95")
    args = p.parse_args(argv)

    metrics = load_jsonl(args.metrics)
    workload = load_jsonl(args.workload)
    if not metrics:
        print("metrics 为空")
        return 2
    gpu_rows = [r for r in metrics if r.get("gpu")]
    if not gpu_rows:
        print("metrics 中没有 gpu 字段")
        return 2

    # 活跃窗口：有 workload 事件时，以首尾事件为准；否则全量 metrics
    if workload:
        t_all = [float(e["wall_time"]) for e in workload if e.get("wall_time") is not None]
        t0, t1 = min(t_all) - 5.0, max(t_all) + 5.0
    else:
        t_all = [float(r["wall_time"]) for r in metrics if r.get("wall_time") is not None]
        t0, t1 = min(t_all), max(t_all)
    active = filter_rows(metrics, t0, t1)
    active_gpu = [r["gpu"] for r in active if r.get("gpu")]

    def series(key: str) -> List[float]:
        return [g.get(key) for g in active_gpu if g.get(key) is not None]

    overall: Dict[str, Any] = {
        "gpu_usage_percent": summarize_series(series("gpu_usage_percent")),
        "gpu_power_mw": summarize_series(series("gpu_power_mw")),
        "gpu_freq_mhz": summarize_series(series("gpu_freq_mhz")),
    }
    power_mw = series("gpu_power_mw")
    usage = series("gpu_usage_percent")
    if args.power_ceiling_w is not None:
        ceiling_mw = args.power_ceiling_w * 1000.0
        ceiling_source = "用户指定"
    else:
        ceiling_mw = overall["gpu_power_mw"]["p95"] or 0.0
        ceiling_source = "本轮活跃窗口 P95"
    busy = [u for u in usage if u >= 95.0]
    full = [u for u, pw in zip(usage, power_mw) if u >= 95.0 and pw >= 0.90 * ceiling_mw]
    overall["busy_fraction"] = len(busy) / len(usage) if usage else 0.0
    overall["full_load_fraction"] = len(full) / len(usage) if usage else 0.0

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, "gpu_utilization_overlay.png")
    make_chart(active, out_png)

    # 分阶段
    windows = parse_stage_windows(workload)
    stages: List[Dict[str, Any]] = []
    for c in sorted(windows):
        t0, t1 = windows[c]
        rows = filter_rows(metrics, t0, t1)
        stage_gpu = [r["gpu"] for r in rows if r.get("gpu")]
        if not stage_gpu:
            continue
        s_usage = [g.get("gpu_usage_percent") for g in stage_gpu if g.get("gpu_usage_percent") is not None]
        s_power = [g.get("gpu_power_mw") for g in stage_gpu if g.get("gpu_power_mw") is not None]
        s_freq = [g.get("gpu_freq_mhz") for g in stage_gpu if g.get("gpu_freq_mhz") is not None]
        busy = [u for u in s_usage if u >= 95.0]
        full = [u for u, pw in zip(s_usage, s_power) if u >= 95.0 and pw >= 0.90 * ceiling_mw]
        stages.append({
            "concurrency": c,
            "n": len(stage_gpu),
            "gpu_usage_percent": summarize_series(s_usage),
            "gpu_power_mw": summarize_series(s_power),
            "gpu_freq_mhz": summarize_series(s_freq),
            "busy_fraction": len(busy) / len(s_usage) if s_usage else 0.0,
            "full_load_fraction": len(full) / len(s_usage) if s_usage else 0.0,
        })

    report = render_report(
        args.run_label, args.metrics, args.workload, ceiling_mw,
        overall, stages, out_png, ceiling_source)
    out_md = os.path.join(args.out_dir, "gpu_utilization_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[done] report={out_md}")
    if os.path.exists(out_png):
        print(f"[done] chart={out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
