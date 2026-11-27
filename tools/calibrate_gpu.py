# -*- coding: utf-8 -*-
"""GPU 算力绑定压测校准：扫 duty → 测得 GPU 频率 / 有效频率利用率。

背景
----
GPUStressor 现在用「小矩阵连乘」的算力绑定内核，理论上 duty↑ → GPU 频率↑ →
effective_utilization(=residency×freq/max_freq) 单调上升。本脚本负责离线标定这条
duty→effective_utilization 曲线，供 ``run_contention_bench`` 把 GPU 档位（目标
effective_utilization）反查成 duty。

依赖
----
读取用户已手动启动的 ``sudo powermetrics ... > ~/powermetrics_qwen38.txt`` 正在持续
写入的文件。本脚本只读该文件新增部分，不额外启动 powermetrics、不覆盖该文件。

流程（对每个 duty）
-------------------
1. 起 stressor（算力绑定，--duration 跑得比测量窗口稍长，确保测量期全程在压）。
2. 等 ramp（约 1s），记录 powermetrics 文件当前大小作为测量起点。
3. 测量窗口内 sleep，记录结束大小。
4. 停 stressor；读新增内容，解析 GPU block 求 freq/residency 均值。
5. effective_utilization = residency × freq / max_freq_mhz。

用法
----
    python tools/calibrate_gpu.py \
        --powermetrics-file ~/powermetrics_qwen38.txt \
        --duties 0.1,0.2,0.4,0.6,0.8,1.0 \
        --seconds-per-duty 20 \
        --max-freq-mhz 1620 \
        --out results/gpu_calibration_clean.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRESSOR = os.path.join(ROOT, "tools", "resource_stressor.py")

sys.path.insert(0, ROOT)
from tools.gpu_metrics_stream import split_blocks, parse_gpu_block  # noqa: E402


def _read_append(path: str, start_size: int) -> str:
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    if size <= start_size:
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(start_size)
        return f.read()


def _mean_std(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n if n > 1 else 0.0
    return {"mean": round(m, 2), "std": round(var ** 0.5, 2), "n": n}


def sample_gpu(path: str, start_size: int,
               max_freq_mhz: Optional[float]) -> Dict[str, Any]:
    """读取文件 `start_size` 之后新增内容，汇总 freq/residency/effective/power。"""
    text = _read_append(path, start_size)
    freqs: List[float] = []
    res: List[float] = []
    eff: List[float] = []
    powers: List[float] = []
    for block in split_blocks(text):
        p = parse_gpu_block(block)
        if not p:
            continue
        freq = p.get("active_freq")
        residency = p.get("active_residency")
        power = p.get("gpu_power_mw")
        if freq is not None:
            freqs.append(float(freq))
        if residency is not None:
            res.append(float(residency))
        if power is not None:
            powers.append(float(power))
        if (max_freq_mhz and max_freq_mhz > 0
                and freq is not None and residency is not None):
            eff.append(residency * freq / max_freq_mhz)
    return {
        "freq_mhz": _mean_std(freqs),
        "residency_percent": _mean_std(res),
        "effective_utilization_percent": _mean_std(eff),
        "power_mw": _mean_std(powers),
    }


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def run_knob(duty: float, mat_iters: int, args: argparse.Namespace,
             log_dir: str, label: str) -> Optional[Dict[str, Any]]:
    """单档：起 stressor（给定 duty + mat_iters）→ 测窗口内读 powermetrics → 停。"""
    status_file = os.path.join(log_dir, f"{label}.jsonl")
    cmd = [args.python, STRESSOR, "--kind", "gpu", "--target", "0",
           "--gpu-duty", f"{duty:.3f}", "--duration", f"{args.seconds_per_duty + 3.0:.1f}",
           "--gpu-mat-size", str(args.mat_size), "--gpu-mat-iters", str(mat_iters),
           "--status-file", status_file]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)
    except OSError as exc:
        print(f"[calibrate] stressor 启动失败 {label}: {exc}", file=sys.stderr)
        return None

    # ramp：给 1s 让 GPU 频率起来，并确保在测量窗口内 stressor 全程在压
    time.sleep(1.0)
    start_size = _file_size(args.powermetrics_file)
    time.sleep(args.seconds_per_duty)
    _ = _file_size(args.powermetrics_file)

    # 停 stressor（正常应已因 --duration 退出，此处兜底）
    if proc.poll() is None:
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass

    sample = sample_gpu(args.powermetrics_file, start_size, args.max_freq_mhz)
    row = {"duty": duty, "mat_iters": mat_iters, **sample}
    print(f"[calibrate] {label} duty={duty:.2f} iters={mat_iters} -> "
          f"freq={sample['freq_mhz']['mean']} MHz "
          f"residency={sample['residency_percent']['mean']}% "
          f"effective={sample['effective_utilization_percent']['mean']}% "
          f"power={sample['power_mw']['mean']} mW",
          flush=True)
    return row


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="GPU 算力绑定压测校准（duty→effective_utilization）")
    p.add_argument("--powermetrics-file",
                   default=os.path.expanduser("~/powermetrics_qwen38.txt"))
    p.add_argument("--duties", default="0.1,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--sweep-iters", default=None,
                   help="扫 mat_iters（连续 duty=1.0）而非扫 duty，如 8,32,128,512,2000")
    p.add_argument("--seconds-per-duty", type=float, default=20.0)
    p.add_argument("--max-freq-mhz", type=float, default=1620.0)
    p.add_argument("--mat-size", type=int, default=256)
    p.add_argument("--mat-iters", type=int, default=32)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out", default="results/gpu_calibration_clean.json")
    args = p.parse_args(argv)

    if not os.path.exists(args.powermetrics_file):
        print(f"[calibrate] powermetrics 文件不存在：{args.powermetrics_file}",
              file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / f"{out_path.stem}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    curve: List[Dict[str, Any]] = []
    if args.sweep_iters:
        # 连续计算（duty=1.0），扫 mat_iters 控制算力强度
        mode = "sweep_iters"
        iters = [int(x) for x in args.sweep_iters.split(",") if x.strip()]
        if not iters:
            print("[calibrate] --sweep-iters 为空", file=sys.stderr)
            return 2
        for it in iters:
            row = run_knob(1.0, it, args, str(log_dir), f"iters_{it}")
            if row is not None:
                curve.append(row)
    else:
        mode = "sweep_duty"
        duties = [float(x) for x in args.duties.split(",") if x.strip()]
        if not duties:
            print("[calibrate] --duties 为空", file=sys.stderr)
            return 2
        for d in duties:
            row = run_knob(float(d), args.mat_iters, args, str(log_dir), f"duty_{d:.2f}")
            if row is not None:
                curve.append(row)

    result = {
        "label": "gpu_calibration",
        "mode": mode,
        "powermetrics_file": args.powermetrics_file,
        "max_freq_mhz": args.max_freq_mhz,
        "mat_size": args.mat_size,
        "mat_iters": args.mat_iters,
        "seconds_per_duty": args.seconds_per_duty,
        "curve": curve,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[calibrate] 完成：{len(curve)} 档 -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())