# -*- coding: utf-8 -*-
"""资源占用竞争评测编排器：按「资源 × 档位」串行跑 监控 + 压测 + 评测。

流程（对每个 (resource, tier)）：
  1. 建单元目录 `<out>/<resource>_<tier>/`
  2. tier > 0 时：起 stressor（后台，写 status-file）→ 轮询等收敛（超时记 NOT_CONVERGED 继续）
  3. 起 monitor（后台，采样 CPU/内存/GPU/带宽落 metrics.jsonl）
  4. 起 bench（前台，固定并发，跑完即止）→ 写逐请求/汇总
  5. 停 stressor → 停 monitor
  6. 从 metrics.jsonl 汇总「四维达成占用向量」，从 bench stage json 取时延/吞吐
  7. 写单元 manifest + 全矩阵 summary.json / summary.md

口径（与 docs/端侧精品模型_资源占用竞争评测_设计方案.md 一致）：
  - 单目标资源串行：一次只把一类资源推到目标档位，其余资源作为副产物随动并全量记录。
  - 内存档位 = 整机总占用（含模型常驻基线），低于基线的档位由 stressor 报 unreachable。
  - GPU/带宽为开环（status-file 立即 converged），实际达成由 monitor 记录。

用法：
  python tools/run_contention_bench.py \
    --out results/contention/qwen38 \
    --resources cpu,memory,gpu,bandwidth \
    --tiers 0,10,20,30,50,60,80 \
    --bench-args "--engine splash-qwen38 --query-source 882 --concurrency 1 --mode stream --limit-per-dataset 5" \
    --monitor-args "--powermetrics-file /tmp/powermetrics.txt --follow-powermetrics" \
    --baseline-mem-gib 22 --theoretical-bandwidth-gb-s 273

  python tools/run_contention_bench.py ... --dry-run   # 只打印矩阵与命令，不执行

安全：只启停本任务自己 spawn 的 stressor/monitor 子进程（精确 Popen 对象），不外放端口、
不删除文件。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRESSOR = os.path.join(ROOT, "tools", "resource_stressor.py")
MONITOR = os.path.join(ROOT, "tools", "monitor_machine.py")
BENCH = os.path.join(ROOT, "tools", "bench_accuracy_concurrency.py")

CONVERGE_TIMEOUT_DEFAULT = 60.0


# ---------------------------------------------------------------------------
# 解析辅助
# ---------------------------------------------------------------------------

def _parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def read_last_jsonl(path: str) -> Optional[Dict[str, Any]]:
    """读 JSONL 最后一行；文件缺失/空/坏行返回 None。"""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        last = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except ValueError:
                continue
        return last


def _mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    return {"mean": round(mean, 3), "std": round(var ** 0.5, 3), "n": n}


def metrics_vector(monitor_dir: str) -> Dict[str, Any]:
    """从 monitor 的 metrics.jsonl 汇总四维占用向量（均值/标准差）。"""
    path = os.path.join(monitor_dir, "metrics.jsonl")
    acc: Dict[str, List[float]] = {
        "cpu_percent": [], "mem_used_percent": [],
        "gpu_usage_percent": [], "bandwidth_utilization": [],
        "gpu_power_mw": [], "gpu_freq_mhz": [],
        "gpu_effective_utilization_percent": [],
    }
    if not os.path.exists(path):
        return {k: _mean_std([]) for k in acc}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            cpu = (row.get("cpu") or {})
            mem = (row.get("memory") or {})
            gpu = (row.get("gpu") or {})
            comp = (row.get("compute") or {})
            if cpu.get("used_percent") is not None:
                acc["cpu_percent"].append(float(cpu["used_percent"]))
            used = mem.get("used_gib_approx")
            total = mem.get("total_gib")
            if used is not None and total:
                acc["mem_used_percent"].append(float(used) / float(total) * 100.0)
            if gpu.get("gpu_usage_percent") is not None:
                acc["gpu_usage_percent"].append(float(gpu["gpu_usage_percent"]))
            if gpu.get("gpu_power_mw") is not None:
                acc["gpu_power_mw"].append(float(gpu["gpu_power_mw"]))
            if gpu.get("gpu_freq_mhz") is not None:
                acc["gpu_freq_mhz"].append(float(gpu["gpu_freq_mhz"]))
            if gpu.get("gpu_effective_utilization_percent") is not None:
                acc["gpu_effective_utilization_percent"].append(
                    float(gpu["gpu_effective_utilization_percent"]))
            bw = comp.get("bandwidth_utilization")
            if bw is not None:
                acc["bandwidth_utilization"].append(float(bw) * 100.0)
    return {k: _mean_std(v) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# 命令构造
# ---------------------------------------------------------------------------

def stressor_cmd(kind: str, tier: int, status_file: str, args: argparse.Namespace
                 ) -> List[str]:
    target = tier
    if kind == "gpu" and args.gpu_mat_iters_list is not None:
        # 连续算力模式：mat_iters 控功耗，--target 仅占位（stressor 强制 target=0）
        target = 0
    cmd = [args.python, STRESSOR, "--kind", kind, "--target", str(target),
           "--duration", "0", "--status-file", status_file]
    if kind == "cpu":
        cmd += ["--reserve-cores", str(args.reserve_cores)]
    elif kind == "memory":
        if args.baseline_mem_gib is not None:
            cmd += ["--baseline-mem-gib", str(args.baseline_mem_gib)]
        cmd += ["--max-memory-percent", str(args.max_memory_percent)]
        if args.memory_extra_gib is not None:
            cmd += ["--extra-gib", str(tier)]
    elif kind == "gpu":
        if args.gpu_mat_iters_list is not None:
            cmd += ["--gpu-duty", "1.0",
                    "--gpu-mat-size", str(args.gpu_mat_size),
                    "--gpu-mat-iters", str(tier)]
        elif args.gpu_duty is not None:
            cmd += ["--gpu-duty", str(args.gpu_duty)]
    elif kind == "bandwidth":
        if args.theoretical_bandwidth_gb_s is not None:
            cmd += ["--theoretical-bandwidth-gb-s", str(args.theoretical_bandwidth_gb_s)]
        if args.calibration_file:
            cmd += ["--calibration-file", args.calibration_file]
        cmd += ["--bandwidth-worker-gb-s", str(args.bandwidth_worker_gb_s)]
    return cmd


def monitor_cmd(unit_dir: str, workload_path: str, args: argparse.Namespace) -> List[str]:
    cmd = [args.python, MONITOR, "--out", os.path.join(unit_dir, "monitor"),
           "--interval", "2", "--duration", "86400"]
    if args.monitor_args:
        cmd += shlex.split(args.monitor_args)
    if workload_path:
        cmd += ["--workload-counters-file", workload_path]
    return cmd


def bench_cmd(unit_dir: str, workload_path: str, args: argparse.Namespace) -> List[str]:
    cmd = [args.python, BENCH] + shlex.split(args.bench_args)
    cmd += ["--out", os.path.join(unit_dir, "bench")]
    if workload_path:
        cmd += ["--live-counters-file", workload_path]
    return cmd


# ---------------------------------------------------------------------------
# 单元执行
# ---------------------------------------------------------------------------

def _start(cmd: List[str], log_path: str) -> subprocess.Popen:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def _stop(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_unit(resource: str, tier: int, args: argparse.Namespace,
             unit_dir: str) -> Dict[str, Any]:
    """执行单个 (resource, tier)，返回单元汇总 dict。"""
    os.makedirs(unit_dir, exist_ok=True)
    workload_path = os.path.join(unit_dir, "workload.jsonl")
    status_file = os.path.join(unit_dir, "stressor_status.jsonl")
    result: Dict[str, Any] = {
        "resource": resource, "tier": tier, "target": float(tier),
        "status": "ok", "achieved_vector": None, "bench": None,
        "stressor_status": None,
    }

    stressor_proc: Optional[subprocess.Popen] = None
    monitor_proc: Optional[subprocess.Popen] = None

    # 1. stressor（tier > 0）
    if tier > 0:
        cmd = stressor_cmd(resource, tier, status_file, args)
        stressor_proc = _start(cmd, os.path.join(unit_dir, "stressor.log"))
        converged, last = _wait_converged(stressor_proc, status_file,
                                          args.converge_timeout)
        result["stressor_status"] = last
        if not converged:
            if last and (last.get("extra") or {}).get("unreachable"):
                result["status"] = "unreachable"
            else:
                result["status"] = "not_converged"
        if result["status"] == "unreachable":
            # 不跑评测
            return result

    # 2. monitor（后台）+ 3. bench（前台）
    monitor_proc = _start(monitor_cmd(unit_dir, workload_path, args),
                          os.path.join(unit_dir, "monitor.log"))
    time.sleep(2.0)  # 让 monitor 起一个采样
    bench_proc = _start(bench_cmd(unit_dir, workload_path, args),
                        os.path.join(unit_dir, "bench.log"))
    bench_rc = bench_proc.wait()

    # 4. 停 monitor + stressor
    _stop(monitor_proc)
    _stop(stressor_proc)

    if bench_rc != 0:
        result["status"] = "bench_failed"

    # 5. 汇总
    result["achieved_vector"] = metrics_vector(os.path.join(unit_dir, "monitor"))
    result["bench"] = read_bench_summary(os.path.join(unit_dir, "bench"))
    return result


def _wait_converged(proc: Optional[subprocess.Popen], status_file: str,
                    timeout: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
    deadline = time.monotonic() + timeout
    last: Optional[Dict[str, Any]] = None
    while time.monotonic() < deadline:
        last = read_last_jsonl(status_file)
        if last is not None and last.get("converged"):
            return True, last
        if proc is not None and proc.poll() is not None:
            # 进程提前退出（如 memory unreachable）
            return False, read_last_jsonl(status_file) or last
        time.sleep(1.0)
    return False, last


def read_bench_summary(bench_dir: str) -> Optional[Dict[str, Any]]:
    """读 bench 的 stage 汇总；无则 None。"""
    files = sorted(glob.glob(os.path.join(bench_dir, "stage_*.json")))
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    keep = ("concurrency", "wall_s", "total_requests", "success_requests",
            "accuracy", "requests_per_s", "tokens_per_s", "input_tokens_per_s",
            "output_tokens_per_s", "ttft_s_mean", "ttft_s_p95",
            "prefill_tok_per_s_mean", "decode_tok_per_s_mean")
    return {k: data.get(k) for k in keep}


# ---------------------------------------------------------------------------
# CLI 与汇总
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_contention_bench",
        description="资源占用竞争评测编排器：资源 × 档位 串行跑 监控+压测+评测")
    p.add_argument("--out", required=True, help="输出根目录（按 <resource>_<tier> 分单元）")
    p.add_argument("--python", default=sys.executable,
                   help="运行 stressor/monitor/bench 的 python 解释器，缺省 sys.executable；"
                        "GPU 需指向装了 mlx 的 venv（如 ~/AA/.venv-314/bin/python3）")
    p.add_argument("--resources", default="cpu,memory,gpu,bandwidth", type=_parse_csv,
                   help="压测资源列表")
    p.add_argument("--tiers", default="0,10,20,30,50,60,80", type=_parse_int_list,
                   help="占用档位百分比")
    p.add_argument("--bench-args", required=True,
                   help="bench_accuracy_concurrency 的完整参数（shlex 解析）")
    p.add_argument("--monitor-args", default="",
                   help="monitor_machine 的额外参数（如 --powermetrics-file ... --follow-powermetrics）")
    p.add_argument("--converge-timeout", type=float, default=CONVERGE_TIMEOUT_DEFAULT,
                   help="等 stressor 收敛超时秒数")
    p.add_argument("--reserve-cores", type=int, default=2)
    p.add_argument("--baseline-mem-gib", type=float, default=None,
                   help="内存基线（GiB）显式覆盖；缺省自动实测整机当前 used_gib")
    p.add_argument("--max-memory-percent", type=float, default=90.0)
    p.add_argument("--memory-extra-gib", default=None,
                   help="内存档位改用「额外分配 GiB」列表（逗号分隔，如 4,8,16），"
                        "覆盖 --tiers 对 memory 的百分比语义；0 表示 baseline")
    p.add_argument("--theoretical-bandwidth-gb-s", type=float, default=None)
    p.add_argument("--bandwidth-worker-gb-s", type=float, default=8.0)
    p.add_argument("--calibration-file", default=None,
                   help="带宽校准曲线 JSON（--calibrate 生成），目标档插值反推 worker 数")
    p.add_argument("--gpu-duty", type=float, default=None)
    p.add_argument("--gpu-mat-iters-list", default=None,
                   help="GPU 档位改用「连续算力 mat_iters」列表（逗号分隔，如 2,8,32,128,512,2000），"
                        "用功耗单调控制；0 表示 baseline。覆盖 --tiers 对 gpu 的百分比语义")
    p.add_argument("--gpu-mat-size", type=int, default=256,
                   help="GPU 算力绑定内核矩阵边长")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印矩阵与命令，不执行")
    return p


def _fmt(v: Any, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def render_summary_md(results: List[Dict[str, Any]]) -> str:
    lines = ["# 资源占用竞争评测汇总\n"]
    lines.append("| 资源 | 档位 | 状态 | CPU% | 内存% | GPU% | 带宽% | "
                 "TTFT均值 | prefill_tps | decode_tps | 吞吐 tok/s | 准确率 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        v = r.get("achieved_vector") or {}
        b = r.get("bench") or {}
        def mean(key):
            return _fmt((v.get(key) or {}).get("mean"))
        lines.append(
            f"| {r['resource']} | {r['tier']} | {r['status']} | {mean('cpu_percent')} "
            f"| {mean('mem_used_percent')} | {mean('gpu_usage_percent')} "
            f"| {mean('bandwidth_utilization')} | {_fmt(b.get('ttft_s_mean'))} "
            f"| {_fmt(b.get('prefill_tok_per_s_mean'))} | {_fmt(b.get('decode_tok_per_s_mean'))} "
            f"| {_fmt(b.get('tokens_per_s'))} "
            f"| {_fmt(b.get('accuracy'))} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    mem_tiers = None
    if args.memory_extra_gib is not None:
        mem_tiers = [0] + _parse_int_list(args.memory_extra_gib)
    gpu_iters = None
    if args.gpu_mat_iters_list is not None:
        gpu_iters = [0] + _parse_int_list(args.gpu_mat_iters_list)
    matrix = []
    for r in args.resources:
        if r == "memory" and mem_tiers is not None:
            tiers = mem_tiers
        elif r == "gpu" and gpu_iters is not None:
            tiers = gpu_iters
        else:
            tiers = sorted(args.tiers)
        for t in tiers:
            matrix.append((r, t))
    if args.dry_run:
        print(f"[dry-run] out={args.out}")
        for resource, tier in matrix:
            unit_dir = os.path.join(args.out, f"{resource}_{tier}")
            status_file = os.path.join(unit_dir, "stressor_status.jsonl")
            print(f"  [{resource:9s} {tier:2d}%] -> {unit_dir}")
            if tier > 0:
                print("    stressor:", " ".join(stressor_cmd(resource, tier, status_file, args)))
            print("    monitor :", " ".join(monitor_cmd(unit_dir, os.path.join(unit_dir, "workload.jsonl"), args)))
            print("    bench   :", " ".join(bench_cmd(unit_dir, os.path.join(unit_dir, "workload.jsonl"), args)))
        return 0

    results: List[Dict[str, Any]] = []
    for resource, tier in matrix:
        unit_dir = os.path.join(args.out, f"{resource}_{tier}")
        print(f"[unit] {resource} {tier}% -> {unit_dir}", flush=True)
        result = run_unit(resource, tier, args, unit_dir)
        results.append(result)
        with open(os.path.join(unit_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  status={result['status']} "
              f"ttft={_fmt((result.get('bench') or {}).get('ttft_s_mean'))}",
              flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(render_summary_md(results))
    print(f"[done] {len(results)} units -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())