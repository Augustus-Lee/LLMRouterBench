# -*- coding: utf-8 -*-
"""资源占用压测器：把 CPU / 内存 / GPU / 带宽推到指定占用档位。

用途
----
在评测模型前，用本脚本在目标机上构造可复现的资源占用（竞争负载），使大模型在
固定占用档位（如 10% / 20% / … / 80%）下被评测。编排器（run_contention_bench）
会在每个档位先启动本脚本、等收敛、再跑 bench。

四类 kind
---------
- ``cpu``       : ``multiprocessing`` 忙等进程，闭环 PID 调进程数，逼近整机 used%
- ``memory``    : ``bytearray`` 分配匿名内存 + keeper 线程触页，整机总占用口径
- ``gpu``       : MLX 小矩阵连乘（算力绑定）+ duty cycle 门控（duty↑→frequency↑）
- ``bandwidth`` : numpy 大数组流式拷贝（释放 GIL），开环按理论带宽反推 worker 数

口径说明（与 docs/端侧精品模型_资源占用竞争评测_设计方案.md 一致）
---------------------------------------------------------------------
- 「单目标资源串行」：一次只把一类资源推到目标档位，其余资源不强控为 0，其实际
  占用作为副产物随动。因此状态上报里的 ``achieved`` 尽量给全量可用读数。
- 内存档位 = 整机总占用（含模型常驻基线）；低于基线的档位输出 ``unreachable``。
- GPU 的 duty→frequency/effective_utilization 映射需用 sudo powermetrics 离线校准，
  实际达成值由 monitor_machine.py 的 ``gpu.gpu_freq_mhz`` / ``gpu_effective_utilization_percent``
  记录。
- 带宽无硬件计数器，属「校准设计目标」，标注 ``calibrated_open_loop``。

用法示例
--------
    python tools/resource_stressor.py --kind cpu --target 50 --duration 60 --status-file /tmp/stress.jsonl
    python tools/resource_stressor.py --kind memory --target 60 --baseline-mem-gib 22 --status-file /tmp/stress.jsonl
    python tools/resource_stressor.py --kind gpu --target 40 --gpu-duty 0.4 --status-file /tmp/stress.jsonl
    python tools/resource_stressor.py --kind bandwidth --target 30 --theoretical-bandwidth-gb-s 273 --status-file /tmp/stress.jsonl
    python tools/resource_stressor.py --kind cpu --target 50 --dry-run

状态文件：每 ``status_interval`` 秒追加一行 JSON：
    {"timestamp": ..., "kind": "cpu", "target": 50, "achieved": {...}, "converged": bool}
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# --- 常量 -----------------------------------------------------------------

CPU_TOLERANCE_PCT = 3.0          # CPU 收敛容差（百分比）
MEM_TOLERANCE_PCT = 3.0          # 内存收敛容差（百分比）
CONVERGE_SAMPLES = 3             # 连续达标采样次数判收敛
PAGE_SIZE = 16384                # Apple Silicon 默认页大小（vm_stat 会覆盖）

_CPU_RE = re.compile(
    r"CPU usage:\s*([\d.]+)%\s+user,\s*([\d.]+)%\s+sys,\s*([\d.]+)%\s+idle",
    re.IGNORECASE,
)


# --- 底层读数（尽力而为，失败返回 None） --------------------------------------


def _run_cmd(cmd: List[str], timeout: float = 10.0) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_percent_psutil() -> Optional[float]:
    try:
        import psutil  # 可选依赖
    except ImportError:
        return None
    try:
        return float(psutil.cpu_percent(interval=0.25))
    except Exception:
        return None


def _cpu_percent_top() -> Optional[float]:
    out = _run_cmd(["top", "-l", "1", "-s", "0", "-n", "0"], timeout=8.0)
    if not out:
        return None
    m = _CPU_RE.search(out)
    if not m:
        return None
    return round(float(m.group(1)) + float(m.group(2)), 2)


def sample_cpu_percent() -> Optional[float]:
    """整机 CPU used%（user+sys）。top 优先（与 monitor_machine 同口径），失败回退 psutil。

    psutil.cpu_percent(interval) 在满核忙等负载下会 0/100 跳变，导致闭环 PID 失控；
    top 的 user+sys 读数在同样负载下稳定，故作为主口径。
    """
    v = _cpu_percent_top()
    return v if v is not None else _cpu_percent_psutil()


def sample_mem_gib() -> Optional[Dict[str, float]]:
    """整机内存 total/used/free(近似) GiB 与 used%。"""
    vm = _run_cmd(["vm_stat"])
    memsize = _run_cmd(["sysctl", "-n", "hw.memsize"])
    if not vm:
        return None
    page_size = PAGE_SIZE
    m = re.search(r"page size of\s+(\d+)\s+bytes", vm)
    if m:
        page_size = int(m.group(1))

    def pages(label: str) -> int:
        mm = re.search(rf"{label}:\s*(\d+)", vm)
        return int(mm.group(1)) if mm else 0

    free = pages("Pages free")
    total_pages = (free + pages("Pages active") + pages("Pages inactive")
                   + pages("Pages speculative") + pages("Pages wired down"))
    def gib(n: int) -> float:
        return round(n * page_size / (1024 ** 3), 4)

    free_gib = gib(free)
    page_total_gib = gib(total_pages)
    # 已用内存 = 页统计和 − 空闲页，与 monitor_machine 的 used_gib_approx 口径一致；
    # 不要用 hw.memsize 当页总数（会高估 ~1~2GiB）。
    used_gib = round(page_total_gib - free_gib, 4)
    total_gib = page_total_gib
    if memsize:
        try:
            total_gib = round(int(memsize.strip()) / (1024 ** 3), 3)
        except ValueError:
            pass
    return {
        "total_gib": total_gib,
        "free_gib": free_gib,
        "used_gib": used_gib,
        "used_percent": round(used_gib / total_gib * 100, 2) if total_gib > 0 else None,
    }


def ncores() -> Optional[int]:
    try:
        return mp.cpu_count()
    except Exception:
        return None


# --- 状态上报 --------------------------------------------------------------


class StatusReporter:
    """把「目标 vs 达成」追加写 JSONL 到 status-file；无文件时仅打印。"""

    def __init__(self, path: Optional[str], kind: str, target: float) -> None:
        self.path = path
        self.kind = kind
        self.target = target
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def report(self, achieved: Dict[str, Any], converged: bool,
               extra: Optional[Dict[str, Any]] = None) -> None:
        row = {
            "timestamp": round(time.time(), 3),
            "kind": self.kind,
            "target": self.target,
            "achieved": achieved,
            "converged": converged,
        }
        if extra:
            row["extra"] = extra
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()


# --- CPU 压测 --------------------------------------------------------------

# busy 子进程自检父进程存活：父进程（stressor）一旦退出成为孤儿（getppid()==1），
# 子进程立即自行退出，从根本上避免被 SIGKILL/中断后残留满核的 while True: pass 进程。
_BUSY_LOOP_CODE = (
    "import os\n"
    "_ppid = os.getppid()\n"
    "while os.getppid() == _ppid:\n"
    "    pass\n"
)


def _spawn_busy() -> "subprocess.Popen[bytes]":
    """启动一个吃满单核的忙等子进程（独立进程，绕开 GIL 与 multiprocessing spawn）。"""
    return subprocess.Popen(
        [sys.executable, "-c", _BUSY_LOOP_CODE],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class CPUStressor:
    def __init__(self, target: float, reserve: int, status: StatusReporter) -> None:
        self.target = target
        self.reserve = max(0, reserve)
        self.status = status
        self.total = ncores() or 1
        self.max_workers = max(1, self.total - self.reserve)
        self.nominal = int(round(self.target / 100.0 * self.total))
        self.nominal = max(0, min(self.nominal, self.max_workers))
        self.procs: List[Any] = []

    def _spawn(self, n: int) -> None:
        for _ in range(n):
            if len(self.procs) >= self.max_workers:
                return
            self.procs.append(_spawn_busy())

    def _kill(self, n: int) -> None:
        for _ in range(n):
            if not self.procs:
                return
            p = self.procs.pop()
            try:
                p.terminate()
            except Exception:
                pass

    def dry_run(self) -> None:
        print(f"[cpu] target={self.target}% ncores={self.total} reserve={self.reserve} "
              f"max_workers={self.max_workers} nominal_workers={self.nominal}")

    def run(self, duration: float) -> None:
        self._spawn(self.nominal)
        started = time.monotonic()
        in_tolerance = 0
        print(f"[cpu] start target={self.target}% workers={self.nominal}/{self.max_workers}",
              flush=True)
        try:
            while duration <= 0 or time.monotonic() - started < duration:
                used = sample_cpu_percent()
                converged = False
                if used is not None:
                    err = self.target - used
                    if abs(err) <= CPU_TOLERANCE_PCT:
                        in_tolerance += 1
                    else:
                        in_tolerance = 0
                        # 简单比例控制：误差换算成需增减的进程数，每次 ±1 避免过冲
                        delta = round(err / 100.0 * self.total)
                        if delta > 0:
                            self._spawn(1)
                        elif delta < 0:
                            self._kill(1)
                    converged = in_tolerance >= CONVERGE_SAMPLES
                    self.status.report({"cpu_percent": used}, converged,
                                       {"workers": len(self.procs)})
                time.sleep(1.0)
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        for p in self.procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=3)
            except Exception:
                pass
        self.procs = []


# --- 内存压测 --------------------------------------------------------------


class MemoryStressor:
    def __init__(self, target: float, baseline_gib: Optional[float], max_pct: float,
                 status: StatusReporter, extra_gib: Optional[float] = None) -> None:
        self.target = target
        self.baseline_gib = baseline_gib  # None = 运行/干跑时实测整机当前 used_gib
        self.max_pct = max_pct
        self.extra_gib = extra_gib  # 非 None 时按「额外分配量」口径，忽略 target 百分比
        self.status = status
        self.buf: Optional[bytearray] = None
        self.allocate_gib: Optional[float] = None
        self._stop = threading.Event()
        self._keeper: Optional[threading.Thread] = None

    def _baseline(self, used_gib: float) -> float:
        return self.baseline_gib if self.baseline_gib is not None else used_gib

    def _compute_allocate(self, total_gib: float, used_gib: float) -> float:
        baseline = self._baseline(used_gib)
        capped_gib = total_gib * self.max_pct / 100.0
        if self.extra_gib is not None:
            # 额外分配量口径：直接分配 extra_gib，但不超过 max_pct 上限
            headroom = capped_gib - baseline
            return round(min(self.extra_gib, max(0.0, headroom)), 3)
        target_gib = min(total_gib * self.target / 100.0, capped_gib)
        return round(target_gib - baseline, 3)

    def dry_run(self) -> None:
        mem = sample_mem_gib()
        if not mem or mem["total_gib"] is None:
            print("[memory] dry-run: 无法读取内存总量")
            return
        total = mem["total_gib"]
        baseline = self._baseline(mem["used_gib"])
        alloc = self._compute_allocate(total, mem["used_gib"])
        if self.extra_gib is not None:
            headroom = total * self.max_pct / 100.0 - baseline
            if alloc <= 0:
                print(f"[memory] unreachable: 额外 {self.extra_gib}GiB 超出 max {self.max_pct}% "
                      f"上限 (headroom={headroom:.1f}GiB, baseline={baseline:.1f}GiB)")
            else:
                print(f"[memory] extra={self.extra_gib}GiB allocate={alloc}GiB "
                      f"(total={total}GiB baseline={baseline:.1f}GiB headroom={headroom:.1f}GiB)")
            return
        if alloc <= 0:
            print(f"[memory] unreachable: 目标 {self.target}% 对应 used={total*self.target/100:.1f}GiB "
                  f"<= 基线 {baseline:.1f}GiB (total {total:.1f}GiB)")
        else:
            print(f"[memory] target={self.target}% total={total}GiB baseline={baseline:.1f}GiB "
                  f"allocate={alloc}GiB")

    def _keeper_loop(self) -> None:
        """周期触页，尽量保持分配内存 resident（脏/活跃），对抗压缩回收。"""
        stride = 4 * 1024 * 1024  # 每 4MiB 触一字节，权衡驻留强度与 keeper 自身 CPU
        while not self._stop.is_set():
            if self.buf is not None:
                for i in range(0, len(self.buf), stride):
                    self.buf[i] ^= 0x01
            self._stop.wait(2.0)

    def run(self, duration: float) -> None:
        mem = sample_mem_gib()
        if not mem or mem["total_gib"] is None:
            print("[memory] 无法读取内存，退出", flush=True)
            self.status.report({}, False, {"error": "no_mem_info"})
            return
        total = mem["total_gib"]
        baseline = self._baseline(mem["used_gib"])
        self.allocate_gib = self._compute_allocate(total, mem["used_gib"])
        if self.allocate_gib <= 0:
            if self.extra_gib is not None:
                print(f"[memory] unreachable: 额外 {self.extra_gib}GiB 超出 max {self.max_pct}% "
                      f"上限 (baseline {baseline:.1f}GiB, total {total}GiB)", flush=True)
                reason = "extra_exceeds_cap"
            else:
                print(f"[memory] unreachable: target {self.target}% 对应 used="
                      f"{total*self.target/100:.1f}GiB <= 基线 {baseline:.1f}GiB "
                      f"(total {total}GiB)", flush=True)
                reason = "target_below_baseline"
            self.status.report({"used_percent": mem["used_percent"]}, False,
                               {"unreachable": True, "reason": reason})
            return
        alloc_bytes = int(self.allocate_gib * (1024 ** 3))
        if self.extra_gib is not None:
            print(f"[memory] extra start allocate={self.allocate_gib}GiB "
                  f"(target extra={self.extra_gib}GiB, total {total}GiB, baseline {baseline:.1f}GiB)",
                  flush=True)
        else:
            print(f"[memory] start target={self.target}% allocate={self.allocate_gib}GiB "
                  f"(total {total}GiB, baseline {baseline:.1f}GiB)", flush=True)
        # 首次全量触页（大分配耗时数秒），随后 keeper 周期刷新
        self.buf = bytearray(alloc_bytes)
        self._touch_all()
        self._keeper = threading.Thread(target=self._keeper_loop, daemon=True)
        self._keeper.start()

        started = time.monotonic()
        in_tolerance = 0
        while duration <= 0 or time.monotonic() - started < duration:
            if self._stop.is_set():
                break
            mem = sample_mem_gib()
            if self.extra_gib is not None:
                # 额外分配量：一次性分配即达成（开环），持续上报 used 变化供监控
                self.status.report(
                    {"used_percent": mem["used_percent"] if mem else None,
                     "used_gib": mem["used_gib"] if mem else None,
                     "extra_allocated_gib": self.allocate_gib},
                    True, {"allocate_gib": self.allocate_gib})
                time.sleep(2.0)
                continue
            converged = False
            if mem and mem["used_percent"] is not None:
                err = self.target - mem["used_percent"]
                if abs(err) <= MEM_TOLERANCE_PCT:
                    in_tolerance += 1
                else:
                    in_tolerance = 0
                converged = in_tolerance >= CONVERGE_SAMPLES
                self.status.report(
                    {"used_percent": mem["used_percent"], "used_gib": mem["used_gib"]},
                    converged, {"allocate_gib": self.allocate_gib})
            time.sleep(2.0)
        self._shutdown()

    def _touch_all(self) -> None:
        if self.buf is None:
            return
        stride = PAGE_SIZE
        for i in range(0, len(self.buf), stride):
            self.buf[i] = 0x01

    def _shutdown(self) -> None:
        self._stop.set()
        if self._keeper is not None:
            self._keeper.join(timeout=3)
        self.buf = None


# --- GPU 压测（MLX matmul + duty cycle） ------------------------------------


class GPUStressor:
    """GPU 压测（默认访存绑定，抢统一内存带宽）。

    默认 2048×2048 大矩阵 matmul，属访存绑定——与模型 decode 抢内存带宽，正好压到
    decode 的真正瓶颈，使 decode/prefill 吞吐随 duty 单调下降（实测 59→30 tok/s）。
    可选算力绑定：调小 ``--gpu-mat-size``（如 256）+ 增大 ``--gpu-mat-iters``，
    用于抬升 GPU 频率/功耗；但模型 decode 不缺算力，算力绑定对吞吐几乎无影响。
    注意 Apple GPU 的 DVFS 对 duty-cycle 是二值响应（空闲/连续），effective_utilization
    不是平滑可控量，仅作旁证记录，不作为档位轴。
    """

    def __init__(self, target: float, duty: Optional[float], status: StatusReporter,
                 mat_size: int = 2048, mat_iters: int = 1) -> None:
        self.target = target
        # duty 一阶近似 = target/100；可由 --gpu-duty 显式覆盖
        self.duty = duty if duty is not None else target / 100.0
        self.duty = max(0.01, min(1.0, self.duty))
        self.mat_size = max(32, int(mat_size))
        self.mat_iters = max(1, int(mat_iters))
        self.status = status
        self._stop = threading.Event()
        self.mx = None

    def _import_mlx(self) -> None:
        try:
            import mlx.core as mx  # noqa: WPS433
            self.mx = mx
        except ImportError:
            self.mx = None

    def dry_run(self) -> None:
        print(f"[gpu] target={self.target}% duty={self.duty:.3f} "
              f"kernel=matmul({self.mat_size}x{self.mat_size})x{self.mat_iters} "
              f"(默认访存绑定抢带宽；调小 mat_size 可切换算力绑定)")

    def run(self, duration: float) -> None:
        self._import_mlx()
        if self.mx is None:
            print("[gpu] MLX 不可用，跳过 GPU 压测", flush=True)
            self.status.report({}, False, {"unavailable": True, "reason": "no_mlx"})
            return
        mx = self.mx
        n = self.mat_size
        a = mx.random.normal(shape=(n, n))
        b = mx.random.normal(shape=(n, n))
        burst_s = 0.05
        idle_s = burst_s * (1.0 - self.duty) / self.duty
        print(f"[gpu] start target={self.target}% duty={self.duty:.3f} "
              f"kernel=matmul({n}x{n})x{self.mat_iters} burst={burst_s*1000:.0f}ms "
              f"idle={idle_s*1000:.0f}ms", flush=True)
        started = time.monotonic()
        while duration <= 0 or time.monotonic() - started < duration:
            if self._stop.is_set():
                break
            # 计算区间：一条 K 次连乘图一次性 eval（K=1 即单次 matmul，摊销内核启动开销）
            t0 = time.monotonic()
            while time.monotonic() - t0 < burst_s and not self._stop.is_set():
                acc = a
                for _ in range(self.mat_iters):
                    acc = acc @ b
                mx.eval(acc)
            self.status.report({"gpu_duty": round(self.duty, 3),
                                "mat_size": n, "mat_iters": self.mat_iters}, True)
            # 空闲区间
            self._stop.wait(idle_s)
        self._shutdown()

    def _shutdown(self) -> None:
        self._stop.set()


# --- 带宽压测（numpy 流式拷贝，自测 GB/s + 校准） -----------------------------


def _stream_worker(stop_evt: threading.Event, gate: Optional[threading.Event],
                   local_counter: List[int], buf_bytes: int) -> None:
    """单 worker：在私有大缓冲上反复读+写，累计搬移字节数。

    ``gate`` 非空时按 duty-cycle 门控：gate.set() 期间满带宽搬移，gate.clear() 时停，
    用于把带宽精确压到低档位（< 单 worker 满带宽的档位）。
    """
    import numpy as np  # noqa: WPS433
    n = max(2, buf_bytes // 8)
    a = np.empty(n, dtype=np.float64)
    half = n // 2
    src = a[:half]
    dst = a[half:]
    per_iter = 3 * half * 8  # 拷贝(读+写) + sum(读)
    while not stop_evt.is_set():
        if gate is not None:
            gate.wait()
        dst[:] = src
        _ = float(src.sum())
        local_counter[0] += per_iter


class BandwidthStressor:
    def __init__(self, target: float, theoretical_gb_s: Optional[float],
                 worker_gb_s: float, workers: Optional[int],
                 status: StatusReporter, buffer_gib: float,
                 calibration_file: Optional[str] = None,
                 calibrate_out: Optional[str] = None) -> None:
        self.target = target
        self.theoretical = theoretical_gb_s
        self.worker_gb_s = worker_gb_s
        self.workers = workers
        self.status = status
        self.buf_bytes = max(8 * 1024 * 1024, int(buffer_gib * (1024 ** 3)))
        self.calibration_file = calibration_file
        self.calibrate_out = calibrate_out
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._counters: List[List[int]] = []
        self.np = None

    def _import_numpy(self) -> None:
        try:
            import numpy  # noqa: WPS433
            self.np = numpy
        except ImportError:
            self.np = None

    def _load_curve(self) -> List[Dict[str, float]]:
        if not self.calibration_file or not os.path.exists(self.calibration_file):
            return []
        try:
            with open(self.calibration_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [p for p in data if isinstance(p, dict)
                    and p.get("workers") and p.get("gb_s")]
        except (ValueError, OSError):
            return []

    def _resolve_target(self) -> Tuple[Optional[int], float]:
        """反推 (workers, duty)：选最小 worker 数使其满带宽覆盖目标，剩余用 duty 门控。

        duty=1.0 表示无需门控（满带宽即目标，或已饱和）。有校准曲线时按曲线满带宽
        算 duty，否则按 --bandwidth-worker-gb-s 线性估算。
        """
        if not self.theoretical:
            return ((max(1, self.workers) if self.workers is not None else None), 1.0)
        target_gb_s = self.theoretical * self.target / 100.0
        curve = self._load_curve()

        def full_gb_s(w: int) -> Optional[float]:
            for p in curve:
                if int(p["workers"]) == w:
                    return float(p["gb_s"])
            return self.worker_gb_s if self.worker_gb_s > 0 else None

        if self.workers is not None:
            w = max(1, self.workers)
            full = full_gb_s(w) or (w * self.worker_gb_s)
            return w, (min(1.0, target_gb_s / full) if full and full > 0 else 1.0)

        if curve:
            pts = sorted(curve, key=lambda p: p["workers"])
            for p in pts:
                if float(p["gb_s"]) >= target_gb_s:
                    return int(p["workers"]), min(1.0, target_gb_s / float(p["gb_s"]))
            last = pts[-1]
            return int(last["workers"]), 1.0

        if self.worker_gb_s > 0:
            w = max(1, int(round(target_gb_s / self.worker_gb_s)))
            return w, min(1.0, target_gb_s / (w * self.worker_gb_s))
        return (None, 1.0)

    def _start(self, wc: int, gate: Optional[threading.Event] = None) -> None:
        self._stop.clear()
        self._counters = [[0] for _ in range(wc)]
        self._threads = [
            threading.Thread(target=_stream_worker,
                             args=(self._stop, gate, c, self.buf_bytes), daemon=True)
            for c in self._counters
        ]
        for t in self._threads:
            t.start()

    def _measure_gbs(self, workers: int, seconds: float) -> float:
        """起 workers 个线程跑 seconds 秒，返回实测 GB/s。"""
        self._start(workers)
        time.sleep(0.3)  # 预热
        prev = sum(c[0] for c in self._counters)
        t0 = time.monotonic()
        time.sleep(seconds)
        total = sum(c[0] for c in self._counters)
        dt = time.monotonic() - t0
        self._shutdown()
        return (total - prev) / 1e9 / dt if dt > 0 else 0.0

    def calibrate(self, seconds: float) -> None:
        self._import_numpy()
        if self.np is None:
            print("[bandwidth] numpy 不可用，无法校准", flush=True)
            return
        total = ncores() or 8
        results: List[Dict[str, Any]] = []
        print(f"[calibrate] 缓冲 {self.buf_bytes // (1024**2)}MiB/worker，"
              f"每档 {seconds}s", flush=True)
        for w in (1, 2, 4, 8, 16):
            if w > total * 2:
                break
            gbs = self._measure_gbs(w, seconds)
            results.append({"workers": w, "gb_s": round(gbs, 2)})
            print(f"[calibrate] workers={w:2d} -> {gbs:7.1f} GB/s", flush=True)
        if results:
            single = results[0]["gb_s"]
            peak = max(r["gb_s"] for r in results)
            print(f"[calibrate] 单 worker={single:.1f} GB/s，峰值≈{peak:.1f} GB/s")
            print(f"[calibrate] 建议 --bandwidth-worker-gb-s {single:.1f}")
        if results and self.calibrate_out:
            with open(self.calibrate_out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[calibrate] 曲线写入 {self.calibrate_out}")

    def dry_run(self) -> None:
        wc, duty = self._resolve_target()
        target_gb_s = (round(self.theoretical * self.target / 100.0, 1)
                       if self.theoretical else None)
        mode = "duty_cycle_open_loop" if duty < 1.0 else "self_measured_open_loop"
        print(f"[bandwidth] target={self.target}% theoretical={self.theoretical}GB/s "
              f"target_gb_s={target_gb_s} workers={wc} duty={duty:.3f} ({mode})")

    def run(self, duration: float) -> None:
        self._import_numpy()
        if self.np is None:
            print("[bandwidth] numpy 不可用，跳过带宽压测", flush=True)
            self.status.report({}, False, {"unavailable": True, "reason": "no_numpy"})
            return
        workers, duty = self._resolve_target()
        if workers is None:
            print("[bandwidth] 需提供 --theoretical-bandwidth-gb-s 或 --bandwidth-workers",
                  flush=True)
            self.status.report({}, False, {"error": "missing_bandwidth_params"})
            return
        gate = threading.Event()
        if duty >= 1.0:
            gate.set()
        self._start(workers, gate=gate)
        burst_s = 0.1
        idle_s = burst_s * (1.0 - duty) / duty if 0.0 < duty < 1.0 else 0.0
        mode = "duty_cycle_open_loop" if duty < 1.0 else "self_measured_open_loop"
        print(f"[bandwidth] start target={self.target}% workers={workers} duty={duty:.3f} "
              f"burst={burst_s * 1000:.0f}ms idle={idle_s * 1000:.0f}ms "
              f"buffer={self.buf_bytes // (1024 ** 2)}MiB/worker ({mode})", flush=True)
        started = time.monotonic()
        prev = 0
        prev_t = time.monotonic()
        try:
            while duration <= 0 or time.monotonic() - started < duration:
                if 0.0 < duty < 1.0:
                    gate.set()
                    self._stop.wait(burst_s)
                    gate.clear()
                    self._stop.wait(idle_s)
                else:
                    self._stop.wait(1.0)
                now = time.monotonic()
                total = sum(c[0] for c in self._counters)
                gbs = (total - prev) / 1e9 / (now - prev_t) if now > prev_t else 0.0
                prev, prev_t = total, now
                self.status.report({"workers": workers, "duty": round(duty, 3),
                                    "measured_gb_s": round(gbs, 2)}, True,
                                   {"measurement": mode})
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)
        self._threads = []
        self._counters = []


# --- CLI -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="resource_stressor",
        description="资源占用压测器：把 CPU/内存/GPU/带宽推到指定占用档位")
    p.add_argument("--kind", required=True, choices=("cpu", "memory", "gpu", "bandwidth"))
    p.add_argument("--target", type=float, default=None,
                   help="目标占用百分比（0~100），整机口径；--calibrate 时无需提供")
    p.add_argument("--duration", type=float, default=0.0,
                   help="压测时长秒；0=持续到被 SIGTERM/SIGINT 结束")
    p.add_argument("--status-file", default=None, help="状态 JSONL 输出路径")
    p.add_argument("--status-interval", type=float, default=1.0)  # 占位，控制各异
    p.add_argument("--reserve-cores", type=int, default=2,
                   help="CPU 预留核数（给监控/评测客户端/server）")
    p.add_argument("--baseline-mem-gib", type=float, default=None,
                   help="内存基线（GiB）显式覆盖；缺省自动实测整机当前 used_gib")
    p.add_argument("--max-memory-percent", type=float, default=90.0,
                   help="内存占用安全上限（百分比，0~100）")
    p.add_argument("--extra-gib", type=float, default=None,
                   help="内存「额外分配量」口径：直接额外分配该 GiB（替代百分比档位）")
    p.add_argument("--gpu-duty", type=float, default=None,
                   help="GPU duty cycle 显式覆盖（0~1）；缺省=target/100")
    p.add_argument("--gpu-mat-size", type=int, default=2048,
                   help="GPU matmul 矩阵边长（默认 2048=访存绑定抢带宽；调小如 256 切算力绑定）")
    p.add_argument("--gpu-mat-iters", type=int, default=1,
                   help="GPU 单次 eval 连乘次数（默认 1=单次 matmul）")
    p.add_argument("--theoretical-bandwidth-gb-s", type=float, default=None,
                   help="理论内存带宽 GB/s（带宽档反推 worker 数用）")
    p.add_argument("--bandwidth-worker-gb-s", type=float, default=8.0,
                   help="单 worker 流式吞吐 GB/s（由 --calibrate 实测得到）")
    p.add_argument("--bandwidth-workers", type=int, default=None,
                   help="带宽 worker 数显式覆盖")
    p.add_argument("--buffer-gib", type=float, default=0.25,
                   help="带宽 worker 单缓冲大小 GiB（需大于系统级缓存）")
    p.add_argument("--calibrate", action="store_true",
                   help="带宽校准：扫描 1/2/4/8/16 worker 实测 GB/s")
    p.add_argument("--calibrate-seconds", type=float, default=3.0,
                   help="校准每档测量秒数")
    p.add_argument("--calibrate-out", default=None,
                   help="校准曲线 JSON 输出路径")
    p.add_argument("--calibration-file", default=None,
                   help="校准曲线 JSON 输入路径（目标档反推 worker 数时插值用）")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不产生负载")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    target = args.target
    if args.kind == "bandwidth" and args.calibrate:
        target = 0.0  # 校准无需 target
    if target is None:
        print("缺少 --target（仅 --calibrate 可省略）", file=sys.stderr)
        return 2
    if not (0.0 <= float(target) <= 100.0):
        print("--target 必须在 0~100 之间", file=sys.stderr)
        return 2

    status = StatusReporter(args.status_file, args.kind, target)
    if args.kind == "cpu":
        s = CPUStressor(target, args.reserve_cores, status)
        if args.dry_run:
            s.dry_run()
            return 0
        s.run(args.duration)
    elif args.kind == "memory":
        s = MemoryStressor(target, args.baseline_mem_gib, args.max_memory_percent, status,
                           extra_gib=args.extra_gib)
        if args.dry_run:
            s.dry_run()
            return 0
        s.run(args.duration)
    elif args.kind == "gpu":
        s = GPUStressor(target, args.gpu_duty, status,
                        mat_size=args.gpu_mat_size, mat_iters=args.gpu_mat_iters)
        if args.dry_run:
            s.dry_run()
            return 0
        s.run(args.duration)
    elif args.kind == "bandwidth":
        s = BandwidthStressor(target, args.theoretical_bandwidth_gb_s,
                              args.bandwidth_worker_gb_s, args.bandwidth_workers,
                              status, args.buffer_gib,
                              calibration_file=args.calibration_file,
                              calibrate_out=args.calibrate_out)
        if args.calibrate:
            s.calibrate(args.calibrate_seconds)
            return 0
        if args.dry_run:
            s.dry_run()
            return 0
        s.run(args.duration)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[stressor] interrupted", flush=True)
        raise SystemExit(130)