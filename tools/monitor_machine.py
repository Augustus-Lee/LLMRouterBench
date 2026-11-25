# -*- coding: utf-8 -*-
"""Mac 模型性能测试机器指标实时监控与图表输出。

监控项
------
- CPU 使用率：system-wide user/sys/idle，总量按 100%，另记录 `ncores`
- 内存：统一内存 free / active / inactive / wired / purgeable / compressor / swap
- SSD：`iostat` 读速率（用于检测权重/专家流式读取）
- GPU / NPU：从外部 CSV 或 `powermetrics` 文本读入；支持跟随正在增长的日志
- 算力 / 带宽：理论基线静态标注；有效算力与有效带宽由实测 token
  吞吐和显式提供的每 token 字节数推导，不把推导值冒充硬件计数器

用法示例
--------
    # 边跑 powermetrics，边实时跟随（推荐在 tmux 中启动 powermetrics）
    sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal \
        -i 2000 -o /tmp/powermetrics.txt
    python tools/monitor_machine.py --duration 3600 --interval 2 \
        --powermetrics-file /tmp/powermetrics.txt --follow-powermetrics \
        --theoretical-tflops 0 --official-bandwidth-gb-s 0 \
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
import copy
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
GPU_FREQ_RE = re.compile(r"GPU HW active frequency:\s*([\d.]+)\s*MHz", re.IGNORECASE)
ANE_FREQ_RE = re.compile(r"ANE HW active frequency:\s*([\d.]+)\s*MHz", re.IGNORECASE)
THERMAL_RE = re.compile(r"Thermal pressure:\s*(\w+)", re.IGNORECASE)
POWER_SAMPLE_SPLIT_RE = re.compile(
    r"(?=^\*\*\* Sampled system activity \([^\n]+\) \*\*\*$)",
    re.MULTILINE,
)


def _first_match(pattern: re.Pattern[str], text: str) -> Optional[float]:
    m = pattern.search(text)
    return float(m.group(1)) if m else None


def split_powermetrics_samples(text: str) -> List[str]:
    """按 powermetrics 的采样块切分；没有块头时按单个样本处理。"""
    parts = [p for p in POWER_SAMPLE_SPLIT_RE.split(text) if p.strip()]
    return parts or ([text] if text.strip() else [])


def parse_powermetrics_sample(text: str) -> Dict[str, Optional[Any]]:
    """解析一个 powermetrics 采样块；缺失字段保留 None，不补 0。"""
    thermal = THERMAL_RE.search(text)
    return {
        "gpu_usage_percent": _first_match(GPU_RES_RE, text),
        "gpu_power_mw": _first_match(GPU_POWER_RE, text),
        "gpu_freq_mhz": _first_match(GPU_FREQ_RE, text),
        "npu_power_mw": _first_match(ANE_POWER_RE, text),
        "npu_usage_percent": _first_match(ANE_RES_RE, text),
        "npu_freq_mhz": _first_match(ANE_FREQ_RE, text),
        "thermal_pressure": thermal.group(1) if thermal else None,
    }


def parse_powermetrics_text(text: str) -> List[Dict[str, Optional[Any]]]:
    """解析 `powermetrics --samplers gpu_power,ane_power,thermal` 文本输出。

    每个 sample block 内分别取首个匹配，避免不同计数器数量不一致时错配。
    无法可靠解析的字段保留 None，不补 0。
    """
    return [parse_powermetrics_sample(block)
            for block in split_powermetrics_samples(text)]


class PowermetricsTail:
    """增量读取 powermetrics 文本文件，并把完整 sample block 交给调用方。

    powermetrics 在追加块时不保证一次 write 就是完整块；只有检测到下一个
    block header 或 EOF 时才消费前一个块。EOF 时若 ``flush=True`` 则消费
    最后一个完整块（用于结束后导入已有文件）。
    """

    def __init__(self, path: str, start_at_end: bool = False) -> None:
        self.path = path
        self._offset = 0
        self._buffer = ""
        self._inode: Optional[int] = None
        if os.path.exists(path):
            self._inode = os.stat(path).st_ino
            if start_at_end:
                self._offset = os.path.getsize(path)

    def read(self, flush: bool = False) -> List[Dict[str, Optional[Any]]]:
        if not os.path.exists(self.path):
            return []
        st = os.stat(self.path)
        if self._inode is not None and st.st_ino != self._inode:
            # 日志被轮转/重建：从头读新文件
            self._offset = 0
            self._buffer = ""
        self._inode = st.st_ino
        if st.st_size < self._offset:
            self._offset = 0
            self._buffer = ""
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(self._offset)
            chunk = f.read()
            self._offset = f.tell()
        self._buffer += chunk
        starts = [m.start() for m in POWER_SAMPLE_SPLIT_RE.finditer(self._buffer)]
        blocks: List[str] = []
        if len(starts) >= 2:
            blocks = [self._buffer[starts[i]:starts[i + 1]]
                      for i in range(len(starts) - 1)]
            self._buffer = self._buffer[starts[-1]:]
        elif flush and self._buffer.strip():
            blocks = [self._buffer]
            self._buffer = ""
        return [parse_powermetrics_sample(b) for b in blocks]


class WorkloadCounterTail:
    """增量读取评测脚本写入的 workload JSONL。

    每行是已完成请求的一个事件。只消费完整行；文件轮转或截断时从头重读。
    坏 JSON 行跳过，避免监控进程影响正在运行的评测。
    """

    def __init__(self, path: str, start_at_end: bool = False) -> None:
        self.path = path
        self._offset = 0
        self._buffer = ""
        self._inode: Optional[int] = None
        if os.path.exists(path):
            self._inode = os.stat(path).st_ino
            if start_at_end:
                self._offset = os.path.getsize(path)

    def read(self, flush: bool = False) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        st = os.stat(self.path)
        if self._inode is not None and st.st_ino != self._inode:
            self._offset = 0
            self._buffer = ""
        self._inode = st.st_ino
        if st.st_size < self._offset:
            self._offset = 0
            self._buffer = ""
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(self._offset)
            chunk = f.read()
            self._offset = f.tell()
        self._buffer += chunk
        lines = self._buffer.split("\n")
        if flush:
            self._buffer = ""
        else:
            self._buffer = lines.pop()
        rows: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                rows.append(event)
        return rows


class WorkloadCounterRing:
    """保存滑动窗口所需事件，并限制内存与陈旧数据。"""

    def __init__(self, window_s: float, max_events: int = 10000) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = window_s
        self.max_events = max(1, max_events)
        self.events: List[Dict[str, Any]] = []

    def add(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def prune(self, now: float) -> None:
        cutoff = now - self.window_s * 2.0
        pruned: List[Dict[str, Any]] = []
        for event in self.events:
            try:
                t = float(event.get("wall_time"))
            except (TypeError, ValueError):
                continue
            if t >= cutoff:
                pruned.append(event)
        self.events = pruned[-self.max_events:]
        if self.events:
            try:
                latest = float(self.events[-1].get("wall_time"))
            except (TypeError, ValueError):
                return
            if latest < cutoff:
                self.events = []


def derive_phase_effective_rates(
        theoretical_tflops: Optional[float],
        theoretical_bandwidth_gb_s: Optional[float],
        prefill_tokens_per_s: Optional[float],
        decode_tokens_per_s: Optional[float],
        prefill_flops_per_token: Optional[float],
        decode_flops_per_token: Optional[float],
        prefill_bytes_per_token: Optional[float],
        decode_bytes_per_token: Optional[float]) -> Dict[str, Optional[float]]:
    """分别由 Prefill/Decode token 吞吐推导分阶段有效算力和带宽。

    不把 Prefill 的 token 数乘 Decode 的 bytes/token，也不将模型总参数
    当作已激活参数量。MoE 的 bytes/token 应使用实际激活专家路径估算。
    """
    prefill = derive_effective_rates(
        theoretical_tflops, theoretical_bandwidth_gb_s,
        prefill_tokens_per_s, prefill_flops_per_token,
        prefill_bytes_per_token)
    decode = derive_effective_rates(
        theoretical_tflops, theoretical_bandwidth_gb_s,
        decode_tokens_per_s, decode_flops_per_token,
        decode_bytes_per_token)
    total_tflops: Optional[float] = None
    total_bandwidth: Optional[float] = None
    if prefill["effective_tflops"] is not None or decode["effective_tflops"] is not None:
        total_tflops = ((prefill["effective_tflops"] or 0.0)
                        + (decode["effective_tflops"] or 0.0))
    if (prefill["effective_bandwidth_gb_s"] is not None
            or decode["effective_bandwidth_gb_s"] is not None):
        total_bandwidth = ((prefill["effective_bandwidth_gb_s"] or 0.0)
                           + (decode["effective_bandwidth_gb_s"] or 0.0))
    return {
        "prefill_effective_tflops": prefill["effective_tflops"],
        "decode_effective_tflops": decode["effective_tflops"],
        "effective_tflops": total_tflops,
        "tflops_utilization": ((total_tflops / theoretical_tflops)
                               if total_tflops is not None and theoretical_tflops
                               and theoretical_tflops > 0 else None),
        "prefill_tflops_utilization": prefill["tflops_utilization"],
        "decode_tflops_utilization": decode["tflops_utilization"],
        "prefill_effective_bandwidth_gb_s": prefill["effective_bandwidth_gb_s"],
        "decode_effective_bandwidth_gb_s": decode["effective_bandwidth_gb_s"],
        "effective_bandwidth_gb_s": total_bandwidth,
        "bandwidth_utilization": ((total_bandwidth / theoretical_bandwidth_gb_s)
                                  if total_bandwidth is not None
                                  and theoretical_bandwidth_gb_s
                                  and theoretical_bandwidth_gb_s > 0 else None),
        "prefill_bandwidth_utilization": prefill["bandwidth_utilization"],
        "decode_bandwidth_utilization": decode["bandwidth_utilization"],
    }


def summarize_workload_window(
        events: List[Dict[str, Any]], now: float, window_s: float,
        theoretical_tflops: Optional[float] = None,
        theoretical_bandwidth_gb_s: Optional[float] = None,
        prefill_flops_per_token: Optional[float] = None,
        decode_flops_per_token: Optional[float] = None,
        prefill_bytes_per_token: Optional[float] = None,
        decode_bytes_per_token: Optional[float] = None) -> Dict[str, Any]:
    """用最近 ``window_s`` 秒内完成请求的 token 数估算实时有效吞吐。

    ``now`` 与事件中的 ``wall_time`` 都必须是 Unix epoch 秒。事件在请求完成
    时写入，因此这是“完成窗口吞吐”，而不是硬件总线计数器；长请求尚未完成
    时可能暂未计入。无有效事件时所有速率与推导值保持 None。
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    valid: List[Dict[str, Any]] = []
    for event in events:
        try:
            t = float(event.get("wall_time"))
        except (TypeError, ValueError):
            continue
        if now - window_s <= t <= now:
            valid.append(event)
    input_known = [e for e in valid if e.get("prompt_tokens") is not None]
    output_known = [e for e in valid if e.get("completion_tokens") is not None]
    input_tokens = sum(float(e.get("prompt_tokens") or 0) for e in input_known)
    output_tokens = sum(float(e.get("completion_tokens") or 0) for e in output_known)
    input_rate = (round(input_tokens / window_s, 2)
                  if valid and len(input_known) == len(valid) else None)
    output_rate = (round(output_tokens / window_s, 2)
                   if valid and len(output_known) == len(valid) else None)
    rates = derive_phase_effective_rates(
        theoretical_tflops, theoretical_bandwidth_gb_s,
        input_rate, output_rate,
        prefill_flops_per_token, decode_flops_per_token,
        prefill_bytes_per_token, decode_bytes_per_token)
    rates.update({
        "measurement": "derived_from_completed_workload_window",
        "window_s": window_s,
        "events_in_window": len(valid),
        "input_tokens_in_window": (input_tokens
                                   if valid and len(input_known) == len(valid) else None),
        "output_tokens_in_window": (output_tokens
                                    if valid and len(output_known) == len(valid) else None),
        "input_tokens_per_s": input_rate,
        "output_tokens_per_s": output_rate,
        "prefill_flops_per_token": prefill_flops_per_token,
        "decode_flops_per_token": decode_flops_per_token,
        "prefill_bytes_per_token": prefill_bytes_per_token,
        "decode_bytes_per_token": decode_bytes_per_token,
    })
    return rates


def derive_effective_rates(theoretical_tflops: Optional[float],
                           theoretical_bandwidth_gb_s: Optional[float],
                           tokens_per_s: Optional[float],
                           flops_per_token: Optional[float],
                           bytes_per_token: Optional[float]) -> Dict[str, Optional[float]]:
    """由实测 token 吞吐推导有效 TFLOPS 和有效带宽。

    这些是 workload-level 的“有效”值，不是硬件总线计数器：
      effective_tflops = tokens/s × FLOPs/token / 1e12
      effective_bandwidth_gb_s = tokens/s × bytes/token / 1e9

    理论值只用于计算利用率。缺少输入时返回 None。
    """
    effective_tflops: Optional[float] = None
    effective_bandwidth: Optional[float] = None
    tflops_util: Optional[float] = None
    bandwidth_util: Optional[float] = None
    if tokens_per_s is not None and flops_per_token is not None:
        effective_tflops = tokens_per_s * flops_per_token / 1e12
        if theoretical_tflops and theoretical_tflops > 0:
            tflops_util = effective_tflops / theoretical_tflops
    if tokens_per_s is not None and bytes_per_token is not None:
        effective_bandwidth = tokens_per_s * bytes_per_token / 1e9
        if theoretical_bandwidth_gb_s and theoretical_bandwidth_gb_s > 0:
            bandwidth_util = effective_bandwidth / theoretical_bandwidth_gb_s
    return {
        "effective_tflops": effective_tflops,
        "tflops_utilization": tflops_util,
        "effective_bandwidth_gb_s": effective_bandwidth,
        "bandwidth_utilization": bandwidth_util,
    }


def build_final_metrics_row(
        last_sample: Optional[Dict[str, Any]],
        latest_gpu: Optional[Dict[str, Any]],
        compute: Dict[str, Any],
        compute_static: Dict[str, Any],
        timestamp: float,
        wall_time: float) -> Dict[str, Any]:
    """以最后一个完整系统采样为基底构造退出汇总行。

    powermetrics 的最终补读只提供 GPU/NPU 字段；退出行仍必须保留最近一次
    完整采样中的 CPU、统一内存、swap 和磁盘字段。返回新对象，避免修改已
    写入 ``metrics.jsonl`` 的历史采样。
    """
    row = copy.deepcopy(last_sample) if last_sample else {}
    row.pop("final_drain", None)
    if latest_gpu is not None:
        row["gpu"] = copy.deepcopy(latest_gpu)
    row["timestamp"] = round(timestamp, 3)
    row["wall_time"] = round(wall_time, 3)
    row["compute"] = copy.deepcopy(compute)
    row["compute_static"] = copy.deepcopy(compute_static)
    row["final_drain"] = True
    return row


def select_compute_snapshot(events_in_window: int,
                            live_compute: Optional[Dict[str, Any]],
                            static_compute: Dict[str, Any]) -> Dict[str, Any]:
    """窗口内有事件时使用实时推导值，否则保留静态基线。"""
    if events_in_window > 0 and live_compute is not None:
        return copy.deepcopy(live_compute)
    return copy.deepcopy(static_compute)


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
        plot(*series(["gpu", "gpu_freq_mhz"]), "GPU frequency (MHz)",
              "gpu_frequency.png", "GPU active frequency over time")
        plot(*series(["gpu", "npu_freq_mhz"]), "NPU frequency (MHz)",
              "npu_frequency.png", "NPU active frequency over time")
    if any(r.get("compute") for r in rows):
        plot(*series(["compute", "prefill_effective_tflops"]), "Prefill effective TFLOPS",
              "prefill_effective_tflops.png", "Prefill effective TFLOPS over time")
        plot(*series(["compute", "decode_effective_tflops"]), "Decode effective TFLOPS",
              "decode_effective_tflops.png", "Decode effective TFLOPS over time")
        plot(*series(["compute", "prefill_effective_bandwidth_gb_s"]),
              "Prefill effective bandwidth (GB/s)",
              "prefill_effective_bandwidth.png", "Prefill effective bandwidth over time")
        plot(*series(["compute", "decode_effective_bandwidth_gb_s"]),
              "Decode effective bandwidth (GB/s)",
              "decode_effective_bandwidth.png", "Decode effective bandwidth over time")
        plot(*series(["compute", "input_tokens_per_s"]), "Input tokens/s",
              "input_tokens_per_s.png", "Prefill token throughput over time")
        plot(*series(["compute", "output_tokens_per_s"]), "Output tokens/s",
              "output_tokens_per_s.png", "Decode token throughput over time")
    return saved


def build_parser() -> argparse.ArgumentParser:
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
                        help="sudo powermetrics 文本输出；自动提取 GPU/NPU/温控")
    parser.add_argument("--follow-powermetrics", action="store_true",
                        help="运行期间持续跟随正在增长的 powermetrics 文件")
    parser.add_argument("--start-at-end", action="store_true",
                        help="跟随已有 powermetrics 文件时只读启动后的新样本")
    parser.add_argument("--powermetrics-interval", type=float, default=1.0,
                        help="powermetrics 采样间隔秒数（仅静态 CSV/旧口径对齐）")
    parser.add_argument("--workload-counters-file", default=None,
                        help="bench 写入的逐请求 workload JSONL；增量跟随并计算实时 token/s")
    parser.add_argument("--workload-window-s", type=float, default=15.0,
                        help="实时吞吐滑动窗口秒数，默认 15")
    parser.add_argument("--start-workload-at-end", action="store_true",
                        help="只消费监控启动后追加的 workload 事件")
    parser.add_argument("--official-bandwidth-gb-s", type=float, default=None,
                        help="硬件理论统一内存带宽 GB/s（静态基线）")
    parser.add_argument("--theoretical-tflops", type=float, default=None,
                        help="硬件理论峰值算力 TFLOPS（静态基线）")
    parser.add_argument("--tokens-per-s", type=float, default=None,
                        help="静态实测 token 吞吐；与逐请求文件同时给出时后者优先")
    parser.add_argument("--flops-per-token", type=float, default=None,
                        help="通用每 token FLOPs；分阶段参数缺省时回退使用")
    parser.add_argument("--bytes-per-token", type=float, default=None,
                        help="通用每 token 字节数；分阶段参数缺省时回退使用")
    parser.add_argument("--prefill-flops-per-token", type=float, default=None,
                        help="Prefill 每输入 token 的有效 FLOPs")
    parser.add_argument("--decode-flops-per-token", type=float, default=None,
                        help="Decode 每输出 token 的有效 FLOPs")
    parser.add_argument("--prefill-bytes-per-token", type=float, default=None,
                        help="Prefill 每输入 token 的有效字节数")
    parser.add_argument("--decode-bytes-per-token", type=float, default=None,
                        help="Decode 每输出 token 的有效字节数；MoE 按激活专家估算")
    parser.add_argument("--quantization", default=None, help="模型量化单位，写入 manifest")
    parser.add_argument("--label", default=None, help="本次监控标签")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

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

    pow_tail: Optional[PowermetricsTail] = None
    if args.powermetrics_file:
        if args.follow_powermetrics:
            pow_tail = PowermetricsTail(args.powermetrics_file,
                                        start_at_end=args.start_at_end)
        elif os.path.exists(args.powermetrics_file):
            with open(args.powermetrics_file, "r", encoding="utf-8") as f:
                pow_samples = parse_powermetrics_text(f.read())
            for i, r in enumerate(pow_samples):
                gpu_by_t[round(i * args.powermetrics_interval, 3)] = r

    prefill_flops = (args.prefill_flops_per_token
                     if args.prefill_flops_per_token is not None
                     else args.flops_per_token)
    decode_flops = (args.decode_flops_per_token
                    if args.decode_flops_per_token is not None
                    else args.flops_per_token)
    prefill_bytes = (args.prefill_bytes_per_token
                     if args.prefill_bytes_per_token is not None
                     else args.bytes_per_token)
    decode_bytes = (args.decode_bytes_per_token
                    if args.decode_bytes_per_token is not None
                    else args.bytes_per_token)
    compute_baseline = derive_phase_effective_rates(
        args.theoretical_tflops, args.official_bandwidth_gb_s,
        args.tokens_per_s, None,
        prefill_flops, decode_flops, prefill_bytes, decode_bytes)
    compute_baseline.update({
        "measurement": "derived_from_static_throughput",
        "tokens_per_s": args.tokens_per_s,
        "flops_per_token": args.flops_per_token,
        "bytes_per_token": args.bytes_per_token,
        "prefill_flops_per_token": prefill_flops,
        "decode_flops_per_token": decode_flops,
        "prefill_bytes_per_token": prefill_bytes,
        "decode_bytes_per_token": decode_bytes,
        "theoretical_tflops": args.theoretical_tflops,
        "official_bandwidth_gb_s": args.official_bandwidth_gb_s,
    })

    manifest = {
        "tool": "monitor_machine",
        "label": args.label,
        "interval_s": args.interval,
        "duration_s": args.duration,
        "disk": args.disk,
        "ncores": ncores,
        "official_bandwidth_gb_s": args.official_bandwidth_gb_s,
        "theoretical_tflops": args.theoretical_tflops,
        "tokens_per_s": args.tokens_per_s,
        "flops_per_token": args.flops_per_token,
        "bytes_per_token": args.bytes_per_token,
        "prefill_flops_per_token": prefill_flops,
        "decode_flops_per_token": decode_flops,
        "prefill_bytes_per_token": prefill_bytes,
        "decode_bytes_per_token": decode_bytes,
        "workload_counters_file": args.workload_counters_file,
        "workload_window_s": args.workload_window_s,
        "start_workload_at_end": args.start_workload_at_end,
        "quantization": args.quantization,
        "gpu_file": args.gpu_file,
        "powermetrics_file": args.powermetrics_file,
        "follow_powermetrics": args.follow_powermetrics,
        "start_at_end": args.start_at_end,
        "powermetrics_interval": args.powermetrics_interval,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("GPU/NPU/温控来自 powermetrics 或外部 CSV；缺失字段为 null。\n"
                 "theoretical_tflops / official_bandwidth_gb_s 是静态硬件基线，"
                 "effective_tflops / effective_bandwidth_gb_s 是由 token 吞吐和"
                 "每 token FLOPs/字节数推导的 workload-level 有效值，不是硬件计数器。"),
        "compute_baseline": compute_baseline,
    }
    write_json(os.path.join(args.out, "manifest.json"), manifest)

    workload_tail: Optional[WorkloadCounterTail] = None
    workload_ring: Optional[WorkloadCounterRing] = None
    if args.workload_counters_file:
        workload_tail = WorkloadCounterTail(
            args.workload_counters_file,
            start_at_end=args.start_workload_at_end)
        workload_ring = WorkloadCounterRing(args.workload_window_s)

    started = time.monotonic()
    idx = 0
    latest_gpu: Optional[Dict[str, Any]] = None
    print(f"[monitor] start duration={args.duration}s interval={args.interval}s "
          f"out={args.out}", flush=True)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    metrics_f = open(metrics_path, "w", encoding="utf-8")
    try:
        while time.monotonic() - started < args.duration:
            t = time.monotonic() - started
            if pow_tail is not None:
                fresh = pow_tail.read(flush=False)
                if fresh:
                    latest_gpu = fresh[-1]
            live_compute: Optional[Dict[str, Any]] = None
            if workload_tail is not None and workload_ring is not None:
                for event in workload_tail.read():
                    workload_ring.add(event)
                now = time.time()
                workload_ring.prune(now)
                live_compute = summarize_workload_window(
                    workload_ring.events, now, args.workload_window_s,
                    args.theoretical_tflops, args.official_bandwidth_gb_s,
                    prefill_flops, decode_flops, prefill_bytes, decode_bytes)
            sample = sample_system(args.disk, ncores)
            sample["timestamp"] = round(t, 3)
            sample["wall_time"] = round(time.time(), 3)
            if latest_gpu is not None:
                sample["gpu"] = dict(latest_gpu)
            else:
                nearest = min(gpu_by_t.keys(), key=lambda x: abs(x - t)) if gpu_by_t else None
                if nearest is not None:
                    g = dict(gpu_by_t[nearest])
                    g.pop("timestamp", None)
                    sample["gpu"] = g
                else:
                    sample["gpu"] = None
            events_in_window = (int((live_compute or {}).get("events_in_window") or 0)
                                if live_compute is not None else 0)
            sample["compute"] = select_compute_snapshot(
                events_in_window, live_compute, compute_baseline)
            sample["compute_static"] = dict(compute_baseline)
            rows.append(sample)
            metrics_f.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
            metrics_f.flush()
            if idx % 10 == 0:
                cpu = sample.get("cpu") or {}
                mem = sample.get("memory") or {}
                gpu = sample.get("gpu") or {}
                print(f"[monitor] t={t:6.1f}s cpu_used={cpu.get('used_percent')}% "
                      f"mem_used={mem.get('used_gib_approx')} GiB "
                      f"gpu={gpu.get('gpu_usage_percent')}", flush=True)
            idx += 1
            time.sleep(max(0.0, min(args.interval, args.duration - (time.monotonic() - started))))
    except KeyboardInterrupt:
        print("[monitor] interrupted by user", flush=True)

    last_sample = copy.deepcopy(rows[-1]) if rows else None
    final_compute: Optional[Dict[str, Any]] = None
    if workload_tail is not None and workload_ring is not None:
        # 退出前补读最后一条已 flush 的事件，避免轮询边界或半行遗留。
        for event in workload_tail.read(flush=True):
            workload_ring.add(event)
        workload_ring.prune(time.time())
        final_compute = summarize_workload_window(
            workload_ring.events, time.time(), args.workload_window_s,
            args.theoretical_tflops, args.official_bandwidth_gb_s,
            prefill_flops, decode_flops, prefill_bytes, decode_bytes)

    drained_gpu = pow_tail.read(flush=True) if pow_tail is not None else []
    if drained_gpu:
        latest_gpu = drained_gpu[-1]
    drained_compute = final_compute if final_compute is not None else compute_baseline
    # 补读到的每个 powermetrics sample 也必须落盘，并把最近完整系统采样作为基底，
    # 避免退出行丢失 CPU、统一内存、swap 或磁盘字段。
    for gpu_sample in drained_gpu:
        drain_row = build_final_metrics_row(
            last_sample, gpu_sample, drained_compute, compute_baseline,
            time.monotonic() - started, time.time())
        rows.append(drain_row)
        metrics_f.write(json.dumps(drain_row, ensure_ascii=False, sort_keys=True) + "\n")
        metrics_f.flush()

    if final_compute is not None:
        # 最终汇总行保留退出时的滑动窗口状态，供图表和阶段汇总使用。
        final_row = build_final_metrics_row(
            last_sample, latest_gpu, final_compute, compute_baseline,
            time.monotonic() - started, time.time())
        rows.append(final_row)
        metrics_f.write(json.dumps(final_row, ensure_ascii=False, sort_keys=True) + "\n")
        metrics_f.flush()
    metrics_f.close()
    charts = make_charts(rows, args.out) if args.chart else []
    print(f"[monitor] done samples={len(rows)} -> {args.out}")
    if charts:
        print(f"[monitor] charts={len(charts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
