# -*- coding: utf-8 -*-
"""流式解析 powermetrics GPU 三行指标并输出 JSONL。

只提取三项核心字段：
  - GPU HW active frequency   -> active_freq
  - GPU HW active residency   -> active_residency
  - GPU Power                 -> gpu_power

有效利用率公式：
  effective_utilization = active_residency * active_freq / max_freq_mhz

默认 max_freq_mhz=1620.0，可通过 --max-freq-mhz 覆盖。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import re

SAMPLE_SPLIT_RE = re.compile(
    r"(?=^\*\*\* Sampled system activity \([^\n]+\) \*\*\*$)",
    re.MULTILINE,
)
GPU_SECTION_RE = re.compile(r"^\*\*\*\s+GPU usage\s+\*\*\*\s*$", re.MULTILINE)
GPU_RES_RE = re.compile(r"GPU HW active residency:\s+([\d.]+)%")
GPU_FREQ_RE = re.compile(r"GPU HW active frequency:\s+(\d+)\s*MHz")
GPU_POWER_RE = re.compile(r"GPU Power:\s+([\d.]+)\s*mW")


def split_blocks(text: str) -> List[str]:
    return [b for b in SAMPLE_SPLIT_RE.split(text) if b.strip()]


def _gpu_section(block: str) -> str:
    m = GPU_SECTION_RE.search(block)
    if not m:
        return block
    start = m.start()
    next_marker = block.find("****", start + 1)
    end = next_marker if next_marker != -1 else len(block)
    return block[start:end]


def parse_gpu_block(block: str) -> Optional[Dict[str, Any]]:
    section = _gpu_section(block)
    m_res = GPU_RES_RE.search(section) or GPU_RES_RE.search(block)
    if not m_res:
        return None

    active_residency = float(m_res.group(1))

    m_freq = GPU_FREQ_RE.search(section) or GPU_FREQ_RE.search(block)
    active_freq = float(m_freq.group(1)) if m_freq else None

    power_matches = GPU_POWER_RE.findall(section)
    if not power_matches:
        power_matches = GPU_POWER_RE.findall(block)
    gpu_power_mw = float(power_matches[-1]) if power_matches else None

    return {
        "active_residency": active_residency,
        "active_freq": active_freq,
        "gpu_power_mw": gpu_power_mw,
    }


def iter_blocks_from_file(path: str) -> Iterator[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    yield from split_blocks(text)


def iter_blocks_from_process(cmd: List[str], duration_s: Optional[float]) -> Iterator[str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr, text=True)
    assert proc.stdout is not None
    started = time.monotonic()
    buffer: List[str] = []
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if SAMPLE_SPLIT_RE.search(line + "\n"):
                if buffer:
                    yield "\n".join(buffer)
                buffer = [line]
            elif buffer:
                buffer.append(line)
            else:
                buffer = [line]

            if duration_s is not None and time.monotonic() - started >= duration_s:
                break
        if buffer:
            yield "\n".join(buffer)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)


def build_row(sample_index: int, p: Dict[str, Any], max_freq_mhz: Optional[float],
              rel_time: float) -> Dict[str, Any]:
    freq_util = None
    effective_util = None
    active_freq = p.get("active_freq")
    if max_freq_mhz and max_freq_mhz > 0 and active_freq is not None:
        freq_util = active_freq / max_freq_mhz * 100.0
        effective_util = p["active_residency"] * active_freq / max_freq_mhz

    gpu = {
        "gpu_usage_percent": p["active_residency"],
        "gpu_freq_mhz": active_freq,
        "gpu_power_mw": p["gpu_power_mw"],
        "gpu_freq_utilization_percent": freq_util,
        "gpu_effective_utilization_percent": effective_util,
        "gpu_max_freq_mhz": max_freq_mhz,
    }
    return {
        "sample_index": sample_index,
        "timestamp": round(rel_time, 3),
        "wall_time": round(time.time(), 3),
        "gpu": gpu,
    }


def write_manifest(path: Path, args: argparse.Namespace, max_freq_mhz: float,
                   sample_count: int) -> None:
    manifest = {
        "tool": "gpu_metrics_stream",
        "label": args.label,
        "source": "powermetrics_file" if args.powermetrics_file else "powermetrics_process",
        "powermetrics_file": args.powermetrics_file,
        "powermetrics_cmd": args.powermetrics_cmd,
        "duration_s": args.duration,
        "max_freq_mhz": max_freq_mhz,
        "sample_count": sample_count,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "effective_gpu_utilization = active_residency * active_freq / max_freq_mhz",
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="流式解析 powermetrics GPU 三行指标")
    p.add_argument("--powermetrics-file", default=None)
    p.add_argument("--powermetrics-cmd",
                   default="sudo -n powermetrics --samplers gpu_power -i 500")
    p.add_argument("--out-jsonl", default="gpu_metrics.jsonl")
    p.add_argument("--manifest", default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--max-freq-mhz", type=float, default=1620.0,
                   help="GPU 固定频率参考上限，默认 1620 MHz")
    p.add_argument("--label", default="gpu-metrics-stream")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if not args.powermetrics_file and not args.powermetrics_cmd:
        print("[gpu_metrics_stream] 必须提供 --powermetrics-file 或 --powermetrics-cmd", file=sys.stderr)
        return 2

    max_freq_mhz = args.max_freq_mhz if args.max_freq_mhz and args.max_freq_mhz > 0 else None
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_path.with_suffix(".manifest.json")

    sample_iter = None  # type: ignore
    start_wall = time.time()
    if args.powermetrics_file:
        sample_iter = ((i, block) for i, block in enumerate(iter_blocks_from_file(args.powermetrics_file), 1))
    else:
        sample_iter = ((i, block) for i, block in enumerate(
            iter_blocks_from_process(args.powermetrics_cmd.split(), args.duration), 1))

    sample_count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sample_index, block in sample_iter:
            parsed = parse_gpu_block(block)
            if not parsed:
                continue
            rel_time = time.time() - start_wall
            row = build_row(sample_index, parsed, max_freq_mhz, rel_time)
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            sample_count += 1
            if not args.quiet:
                g = row["gpu"]
                print(
                    f"[gpu] sample={sample_count} residency={g['gpu_usage_percent']:.1f}% "
                    f"active_freq={g['gpu_freq_mhz']}MHz "
                    f"effective={g['gpu_effective_utilization_percent']}%",
                    flush=True,
                )

    write_manifest(manifest_path, args, max_freq_mhz if max_freq_mhz is not None else 0.0, sample_count)
    print(f"[gpu_metrics_stream] samples={sample_count} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
