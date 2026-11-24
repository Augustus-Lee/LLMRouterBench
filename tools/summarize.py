# -*- coding: utf-8 -*-
"""评测结果汇总：分引擎 Markdown 表（方案 8.2 节）+ 可选图表（8.3 节）。

用法：

    python tools/summarize.py \
        --raw results/mac_benchmark/raw \
        --out results/mac_benchmark

口径约束（v4 强制）：
- 表 A（Splash / Qwen3.8）、表 B（Splash / Qwen3.6）、表 C（MLX-VLM / Gemma）
  各自独立成表，同引擎内比较，跨引擎只做观察不做排名；
- 图表输出在 matplotlib 缺失时自动跳过，表格始终可用。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:  # 支持包导入与脚本直跑两种方式
    from tools.bench_common import summarize_repeats
except ImportError:
    from bench_common import summarize_repeats  # type: ignore

ENGINE_DISPLAY = {"splash": "Splash", "mlx-vlm": "MLX-VLM"}

TABLE_HEADER = (
    "| 上下文 | cold load (s) | TTFT (s) | Prefill tok/s | Decode tok/s "
    "| 峰值 RSS (GiB) | wired (GiB) | swap (GiB) | SSD 读 (MiB/s) | 成功 |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
)


# ---------------------------------------------------------------------------
# 加载与分组
# ---------------------------------------------------------------------------

def load_records(raw_dir: str) -> List[Dict[str, Any]]:
    """读取 raw 目录下全部 JSON 记录（含子目录）。"""
    records: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "**", "*.json"), recursive=True)):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError) as exc:
            print(f"[summarize] skip invalid {path}: {exc}")
            continue
        if isinstance(rec, dict) and "engine" in rec and "repeats" in rec:
            rec["_path"] = path
            records.append(rec)
    return records


def engine_table_letter(model: str) -> str:
    """模型名 -> 方案表号：Qwen3.8=A，Qwen3.6=B，Gemma=C，其余=extra。"""
    m = model.lower()
    if "qwen3.8" in m:
        return "A"
    if "qwen3.6" in m:
        return "B"
    if "gemma" in m:
        return "C"
    return "extra"


def _fmt(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _repeat_field(rec: Dict[str, Any], field: str) -> List[float]:
    return [e[field] for e in rec.get("repeats", [])
            if e.get(field) is not None]


# ---------------------------------------------------------------------------
# 表格渲染
# ---------------------------------------------------------------------------

def render_table(engine: str, records: List[Dict[str, Any]]) -> str:
    """渲染单引擎 Markdown 表；数值取 repeats 均值，None 显示 '-'。"""
    display = ENGINE_DISPLAY.get(engine, engine)
    title = f"### {display}（engine={engine}）\n\n"
    lines = [title, TABLE_HEADER]
    for rec in sorted(records, key=lambda r: r.get("context_tokens", 0)):
        ttft = summarize_repeats(_repeat_field(rec, "ttft_s"))
        prefill = summarize_repeats(_repeat_field(rec, "prefill_tok_per_s"))
        decode = summarize_repeats(_repeat_field(rec, "decode_tok_per_s"))
        mem = rec.get("memory") or {}
        ctx = rec.get("context_tokens", 0)
        lines.append(
            f"| {ctx} "
            f"| {_fmt(rec.get('cold_load_s'))} "
            f"| {_fmt(ttft['mean'], 3)} "
            f"| {_fmt(prefill['mean'])} "
            f"| {_fmt(decode['mean'])} "
            f"| {_fmt(mem.get('peak_process_gib'), 2)} "
            f"| {_fmt(mem.get('wired_gib'), 2)} "
            f"| {_fmt(mem.get('swap_delta_gib'), 3)} "
            f"| {_fmt(rec.get('ssd_read_mib_s'), 1)} "
            f"| {'✅' if rec.get('success') else '❌'} |\n"
        )
        # 每档补一行统计明细（P95），失败/异常样本不删除
        detail = (
            f"  - ctx={ctx}: repeats n={decode['n']}, "
            f"decode mean={_fmt(decode['mean'])} p95={_fmt(decode['p95'])} "
            f"max={_fmt(decode['max'])}, errors={rec.get('error_count', 0)}"
            f", finish={ {e.get('finish_reason') for e in rec.get('repeats', []) if e.get('finish_reason')} or '-'}\n"
        )
        lines.append(detail)
    return "".join(lines)


def render_summary(records: List[Dict[str, Any]]) -> str:
    """汇总文档：按引擎分表 + 跨引擎观察说明（不排名）。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        "# Mac 文本模型性能评测汇总\n\n",
        f"> 生成时间：{now}  \n",
        "> 口径：分引擎记录，同引擎内比较，跨引擎只做观察不做排名"
        "（docs/MAC_文本模型性能评测方案.md v4 第 6 节）。\n\n",
    ]
    engines = sorted({r.get("engine", "unknown") for r in records})
    letters = []
    for engine in engines:
        group = [r for r in records if r.get("engine") == engine]
        model_tags = sorted({r.get("model", "?") for r in group})
        for model_tag in model_tags:
            letters.append(engine_table_letter(model_tag))
        parts.append(f"## 表 {'/'.join(sorted(set(letters)))} — {ENGINE_DISPLAY.get(engine, engine)}\n\n")
        for model_tag in model_tags:
            sub = [r for r in group if r.get("model") == model_tag]
            parts.append(f"**模型：`{model_tag}`**\n\n")
            parts.append(render_table(engine, sub))
        letters = []
    parts.append(
        "\n## 跨引擎观察\n\n"
        "- Splash（DFlash2 投机解码）与 MLX-VLM（普通 decode）的 tok/s "
        "不具直接可比性，仅可对比 TTFT 分布与内存特征；\n"
        "- GGUF Q4_K_M 与 MLX 4bit affine 属不同量化，不可混表；\n"
        "- 失败记录保留原样（error_count > 0 的档位见上表明细）。\n"
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# 图表（matplotlib 可选）
# ---------------------------------------------------------------------------

def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    stats = summarize_repeats(values)
    return stats["mean"]


def _group_series(records: List[Dict[str, Any]], field: str):
    """按 (engine, model) 分组的 (context, mean_value) 序列。"""
    series: Dict[tuple, List[tuple]] = {}
    for rec in sorted(records, key=lambda r: r.get("context_tokens", 0)):
        val = _mean_or_none(_repeat_field(rec, field))
        if val is None:
            continue
        key = (rec.get("engine", "?"), rec.get("model", "?"))
        series.setdefault(key, []).append((rec.get("context_tokens", 0), val))
    return series


def make_charts(records: List[Dict[str, Any]], out_dir: str) -> List[str]:
    """生成方案 8.3 节的 5 张图；matplotlib 缺失或无数据时跳过。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[summarize] matplotlib 不可用，跳过图表输出")
        return []
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    saved: List[str] = []
    colors = {}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    def _color(key):
        if key not in colors:
            colors[key] = palette[len(colors) % len(palette)]
        return colors[key]

    def _plot(series, ylabel, fname, title):
        if not series:
            return
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (engine, model), points in series.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            label = f"{engine}:{model[:40]}"
            ax.plot(xs, ys, marker="o", label=label, color=_color(label))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context tokens")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(charts_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # 图 1：TTFT 随上下文变化（分引擎曲线）
    _plot(_group_series(records, "ttft_s"), "TTFT (s)",
          "chart1_ttft_vs_context.png", "TTFT vs input length")
    # 图 2：Decode tok/s 随上下文变化（分引擎曲线）
    _plot(_group_series(records, "decode_tok_per_s"), "Decode tok/s",
          "chart2_decode_vs_context.png", "Decode throughput vs input length")
    # 图 3：峰值内存随上下文变化
    memory_series: Dict[tuple, List[tuple]] = {}
    for rec in sorted(records, key=lambda r: r.get("context_tokens", 0)):
        mem = rec.get("memory") or {}
        rss = mem.get("peak_process_gib")
        if rss is None:
            continue
        key = (rec.get("engine", "?"), rec.get("model", "?"))
        memory_series.setdefault(key, []).append((rec.get("context_tokens", 0), rss))
    _plot(memory_series, "Peak RSS (GiB)",
          "chart3_memory_vs_context.png", "Peak process RSS vs context")
    # 图 4：同引擎（Splash）内模型对比 —— 由图 2 数据按 model 拆分曲线覆盖
    splash_records = [r for r in records if r.get("engine") == "splash"]
    if splash_records:
        _plot(_group_series(splash_records, "decode_tok_per_s"), "Decode tok/s",
              "chart4_splash_model_compare.png",
              "Splash: same-engine model comparison (A vs B)")
    # 图 5：SSD 读取速率随上下文变化（流式读取检测）
    ssd_series: Dict[tuple, List[tuple]] = {}
    for rec in sorted(records, key=lambda r: r.get("context_tokens", 0)):
        ssd = rec.get("ssd_read_mib_s")
        if ssd is None:
            continue
        key = (rec.get("engine", "?"), rec.get("model", "?"))
        ssd_series.setdefault(key, []).append((rec.get("context_tokens", 0), ssd))
    _plot(ssd_series, "Disk (MiB/s)",
          "chart5_ssd_read_vs_context.png", "SSD activity vs context (streaming detection)")
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="summarize", description="汇总 Mac 性能评测结果（分引擎表 + 图表）")
    parser.add_argument("--raw", default="results/mac_benchmark/raw")
    parser.add_argument("--out", default="results/mac_benchmark")
    args = parser.parse_args(argv)

    records = load_records(args.raw)
    if not records:
        print(f"[summarize] no records found in {args.raw}")
        return 1

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(render_summary(records))
    print(f"[summarize] wrote {summary_path}")

    charts = make_charts(records, args.out)
    for path in charts:
        print(f"[summarize] chart -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
