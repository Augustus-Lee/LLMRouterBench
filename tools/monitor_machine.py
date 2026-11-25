# -*- coding: utf-8 -*-
"""Mac 模型性能测试机器指标实时监控与图表输出。

监控项
------
- CPU 使用率：system-wide user/sys/idle，总量按 100%，另记录 `ncores`
- 内存：统一内存 free / active / inactive / wired / purgeable / compressor / swap
- SSD：`iostat` 读速率（用于检测权重/专家流式读取）
- GPU / NPU：可选，从外部 CSV 时序文件读入（GUI 或 `powermetrics` 预处理产物）

用法示例
--------
    python tools/monitor_machine.py --duration 300 --interval 1 \
        --out results/mac_benchmark/monitor --chart

    # 只测一次确认环境
    python tools/monitor_machine.py --dry-run

GPU/NPU 时序文件（可选 CSV，表头必须存在）：
    timestamp,gpu_usage_percent,npu_usage_percent
    0,12.5,3.2
    1,15.0,3.4

输出：`<out>/metrics.jsonl`、`<out>/manifest.json`，可选 PNG 图表。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

CPU_RE = re.compile(
    r"CPU usage:\s*([\d.]+)%\s+user,\s*([\d.]+)%\s+sys,\s*([\d.]+)%\s+idle",
    re.IGNORECASE,
)
SWAP_RE = re.compile(
    r"total:\s*([\d.]+)\s*([MG])i?B?\s+"
    r"used:\s*([\d.]+)\s*([MG])i?B?\s+"
    r"free:\s*([\d.]+)\s*([MG])i?B?",
    re.IGNORECASE,
)


def _run_cmd(cmd: List[str], timeout: float = 10.0) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
        if out.returncode == 0:
            return out.stdout
        return None
    except (OSError, subprocess.SubprocessError):
        return None


def _to_mib(value: float, unit: str) -> float:
    return value * 1024.0 if unit.upper() == "G" else value


def parse_swapusage(text: str) -> Optional[Dict[str, float]]:
    m = SWAP_RE.search(text)
    if not m:
        return None
    return {
        "total_mib": _to_mib(float(m.group(1)), m.group(2)),
        "used_mib": _to_mib(float(m.group(3)), m.group(4)),
        "free_mib": _to_mib(float(m.group(5)), m.group(6)),
    }


def parse_vm_stat(text: str) -> Dict[str, float]:
    """解析 vm_stat，数量统一换算为 GiB。"""
    page_size = 16384
    m = re.search(r"page size of\s+(\d+)\s+bytes", text)
    if m:
        page_size = int(m.group(1))

    def pages(label: str) -> int:
        mm = re.search(rf"{label}:\s*(\d+)", text)
        return int(mm.group(1)) if mm else 0

    def gib(n: int) -> float:
        return round(n * page_size / (1024 ** 3), 4)

    total_pages = pages("Pages free") + pages("Pages active") + pages("Pages inactive") + \
        pages("Pages speculative") + pages("Pages wired down")
    raw = {
        "page_size": page_size,
        "free_gib": gib(pages("Pages free")),
        "active_gib": gib(pages("Pages active")),
        "inactive_gib": gib(pages("Pages inactive")),
        "speculative_gib": gib(pages("Pages speculative")),
        "wired_gib": gib(pages("Pages wired down")),
        "purgeable_gib": gib(pages("Pages purgeable")),
        "compressor_occupied_gib": gib(pages("Pages occupied by compressor")),
        "compressor_stored_gib": gib(pages("Pages stored in compressor")),
        "file_backed_gib": gib(pages("File-backed pages")),
        "anonymous_gib": gib(pages("Anonymous pages")),
    }
    raw["total_gib_approx"] = gib(total_pages)
    raw["used_gib_approx"] = round(raw["total_gib_approx"] - raw["free_gib"], 4)
    raw["available_gib_approx"] = round(
        raw["free_gib"] + raw["inactive_gib"] + raw["purgeable_gib"], 4)
    return raw


def parse_memsize(text: str) -> Optional[float]:
    if not text:
        return None
    try:
        return round(int(text.strip()) / (1024 ** 3), 1)
    except ValueError:
        return None


def sample_system(disk: Optional[str], ncores: Optional[int]) -> Dict[str, Any]:
    sample: Dict[str, Any] = {}

    top = _run_cmd(["top", "-l", "1", "-s", "0", "-n", "0"], timeout=8.0)
    if top:
        m = CPU_RE.search(top)
        if m:
            user, sysv, idle = float(m.group(1)), float(m.group(2)), float(m.group(3))
            sample["cpu"] = {
                "user_percent": user,
                "sys_percent": sysv,
                "idle_percent": idle,
                "used_percent": round(user + sysv, 2),
                "total_percent": 100.0,
                "ncores": ncores,
            }
        else:
            sample["cpu"] = None
    else:
        sample["cpu"] = None

    vm = _run_cmd(["vm_stat"])
    sample["memory"] = parse_vm_stat(vm) if vm else None

    memsize = _run_cmd(["sysctl", "-n", "hw.memsize"])
    if sample["memory"] and memsize:
        sample["memory"]["total_gib"] = parse_memsize(memsize)

    swap = _run_cmd(["sysctl", "-n", "vm.swapusage"])
    sample["swap"] = parse_swapusage(swap) if swap else None

    if disk:
        disk_text = _run_cmd(["iostat", "-c", "1", "-w", "1", "-d", disk], timeout=8.0)
        if disk_text:
            value = None
            for line in disk_text.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        value = float(parts[-1])
                    except ValueError:
                        continue
            sample["disk_read_mib_s"] = value
        else:
            sample["disk_read_mib_s"] = None
    else:
        sample["disk_read_mib_s"] = None

    return sample


def load_gpu_series(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: Dict[str, float] = {"timestamp": float(row.get("timestamp") or 0.0)}
            for key in ("gpu_usage_percent", "npu_usage_percent"):
                raw = row.get(key)
                out[key] = float(raw) if raw not in (None, "") else None
            rows.append(out)
    return rows


GPU_RES_RE = re.compile(r"GPU active residency:\s*([\d.]+)%")
GPU_POWER_RE = re.compile(r"GPU Power:\s*([\d.]+)\s*mW", re.IGNORECASE)
ANE_POWER_RE = re.compile(r"ANE Power:\s*([\d.]+)\s*mW", re.IGNORECASE)
ANE_RES_RE = re.compile(r"ANE (?:active residency|duty cycle):\s*([\d.]+)%", re.IGNORECASE)


def parse_powermetrics_text(text: str) -> List[Dict[str, Optional[float]]]:
    """解析 `powermetrics --samplers gpu_power,ane_power` 文本输出。

    按出现顺序分别收集 GPU/ANE 指标，再按索引合并为同一样本。
    无法可靠解析的字段保留 None，不补 0。
    """
    gpu_res = [float(x) for x in GPU_RES_RE.findall(text)]
    gpu_power = [float(x) for x in GPU_POWER_RE.findall(text)]
    ane_power = [float(x) for x in ANE_POWER_RE.findall(text)]
    ane_res = [float(x) for x in ANE_RES_RE.findall(text)]
    n = max(len(gpu_res), len(gpu_power), len(ane_power), len(ane_res))
    samples: List[Dict[str, Optional[float]]] = []
    for i in range(n):
        samples.append({
            "gpu_usage_percent": gpu_res[i] if i < len(gpu_res) else None,
            "gpu_power_mw": gpu_power[i] if i < len(gpu_power) else None,
            "npu_power_mw": ane_power[i] if i < len(ane_power) else None,
            "npu_usage_percent": ane_res[i] if i < len(ane_res) else None,
        })
    return samples


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def make_charts(rows: List[Dict[str, Any]], out_dir: str) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[monitor] matplotlib 不可用，跳过图表输出")
        return []
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    saved: List[str] = []

    def series(field_path: List[str]):
        xs, ys = [], []
        for i, row in enumerate(rows):
            v = row
            for key in field_path:
                if not isinstance(v, dict):
                    v = None
                    break
                v = v.get(key)
            if v is None:
                continue
            xs.append(i)
            ys.append(float(v))
        return xs, ys

    def plot(xs, ys, ylabel, fname, title):
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, linewidth=1)
        ax.set_xlabel("sample index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(charts_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    plot(*series(["cpu", "used_percent"]), "CPU used (%)",
          "cpu_usage.png", "CPU usage over time")
    plot(*series(["cpu", "idle_percent"]), "CPU idle (%)",
          "cpu_idle.png", "CPU idle over time")
    plot(*series(["memory", "used_gib_approx"]), "Memory used (GiB)",
          "memory_used.png", "Memory used over time")
    plot(*series(["memory", "free_gib"]), "Memory free (GiB)",
          "memory_free.png", "Memory free over time")
    plot(*series(["memory", "inactive_gib"]), "Memory inactive (GiB)",
          "memory_inactive.png", "Memory inactive over time")
    if any(r.get("swap") for r in rows):
        plot(*series(["swap", "used_mib"]), "Swap used (MiB)",
              "swap_used.png", "Swap used over time")
    if any(r.get("disk_read_mib_s") is not None for r in rows):
        plot(*series(["disk_read_mib_s"]), "Disk read (MiB/s)",
              "disk_read.png", "Disk read over time")
    if any(r.get("gpu") for r in rows):
        plot(*series(["gpu", "gpu_usage_percent"]), "GPU usage (%)",
              "gpu_usage.png", "GPU usage over time")
        plot(*series(["gpu", "npu_usage_percent"]), "NPU usage (%)",
              "npu_usage.png", "NPU usage over time")
        plot(*series(["gpu", "gpu_power_mw"]), "GPU power (mW)",
              "gpu_power.png", "GPU power over time")
        plot(*series(["gpu", "npu_power_mw"]), "NPU power (mW)",
              "npu_power.png", "NPU power over time")
    return saved


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="monitor_machine",
        description="Mac 性能测试机器指标实时监控与图表")
    parser.add_argument("--out", default="results/mac_benchmark/monitor")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--disk", default=None,
                        help="可选 disk 名（如 disk0）；不传则跳过 iostat 采样，略快")
    parser.add_argument("--chart", action="store_true", help="输出 PNG 图表")
    parser.add_argument("--gpu-file", default=None,
                        help="外部 GPU/NPU CSV 时序文件：timestamp,gpu_usage_percent,npu_usage_percent")
    parser.add_argument("--powermetrics-file", default=None,
                        help="sudo powermetrics 文本输出；自动提取 GPU/NPU 使用率与功率")
    parser.add_argument("--powermetrics-interval", type=float, default=1.0,
                        help="powermetrics 采样间隔秒数，用于对齐 monitor 时间轴")
    parser.add_argument("--official-bandwidth-gb-s", type=float, default=None,
                        help="用于 manifest 的 GPU/NPU 官方理论带宽")
    parser.add_argument("--theoretical-tflops", type=float, default=None,
                        help="用于 manifest 的理论峰值算力")
    parser.add_argument("--quantization", default=None, help="模型量化单位，写入 manifest")
    parser.add_argument("--label", default=None, help="本次监控标签")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if platform.system() != "Darwin":
        print("[monitor] warning: 非 macOS 机器，CPU/memory 指标可能为空", file=sys.stderr)

    ncores: Optional[int] = None
    if platform.system() == "Darwin":
        out = _run_cmd(["sysctl", "-n", "hw.ncpu"])
        if out:
            try:
                ncores = int(out.strip())
            except ValueError:
                ncores = None

    if args.dry_run:
        sample = sample_system(args.disk, ncores)
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return 0

    rows: List[Dict[str, Any]] = []
    gpu_by_t: Dict[float, Dict[str, Any]] = {}
    if args.gpu_file:
        for r in load_gpu_series(args.gpu_file):
            gpu_by_t[round(float(r.get("timestamp") or 0.0), 3)] = r
    if args.powermetrics_file:
        with open(args.powermetrics_file, "r", encoding="utf-8") as f:
            pow_samples = parse_powermetrics_text(f.read())
        for i, r in enumerate(pow_samples):
            gpu_by_t[round(i * args.powermetrics_interval, 3)] = r

    manifest = {
        "tool": "monitor_machine",
        "label": args.label,
        "interval_s": args.interval,
        "duration_s": args.duration,
        "disk": args.disk,
        "ncores": ncores,
        "official_bandwidth_gb_s": args.official_bandwidth_gb_s,
        "theoretical_tflops": args.theoretical_tflops,
        "quantization": args.quantization,
        "gpu_file": args.gpu_file,
        "powermetrics_file": args.powermetrics_file,
        "powermetrics_interval": args.powermetrics_interval,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("GPU/NPU 使用率依赖外部时序文件；无 gpu-file 时字段为 null。"
                 "算力/带宽为静态标注，不通过本脚本实测伪造。"),
    }
    write_json(os.path.join(args.out, "manifest.json"), manifest)

    started = time.monotonic()
    idx = 0
    print(f"[monitor] start duration={args.duration}s interval={args.interval}s "
          f"out={args.out}", flush=True)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    metrics_f = open(metrics_path, "w", encoding="utf-8")
    try:
        while time.monotonic() - started < args.duration:
            t = time.monotonic() - started
            sample = sample_system(args.disk, ncores)
            sample["timestamp"] = round(t, 3)
            nearest = min(gpu_by_t.keys(), key=lambda x: abs(x - t)) if gpu_by_t else None
            if nearest is not None:
                g = dict(gpu_by_t[nearest])
                g.pop("timestamp", None)
                sample["gpu"] = g
            else:
                sample["gpu"] = None
            rows.append(sample)
            metrics_f.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
            metrics_f.flush()
            if idx % 10 == 0:
                cpu = sample.get("cpu") or {}
                mem = sample.get("memory") or {}
                print(f"[monitor] t={t:6.1f}s cpu_used={cpu.get('used_percent')}% "
                      f"mem_used={mem.get('used_gib_approx')} GiB", flush=True)
            idx += 1
            time.sleep(max(0.0, min(args.interval, args.duration - (time.monotonic() - started))))
    except KeyboardInterrupt:
        print("[monitor] interrupted by user", flush=True)

    metrics_f.close()
    charts = make_charts(rows, args.out) if args.chart else []
    print(f"[monitor] done samples={len(rows)} -> {args.out}")
    if charts:
        print(f"[monitor] charts={len(charts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
