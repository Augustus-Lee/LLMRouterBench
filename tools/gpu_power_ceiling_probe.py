# -*- coding: utf-8 -*-
"""GPU 压载脚本：用于测量 jiuwen M5 Pro GPU 的功耗上限。

只做重复大矩阵乘法和简单的逐元素运算，尽量占用 GPU 计算与片上访存。
功耗采样必须由外部 powermetrics 同时进行：

    sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal \
        -i 500 -b 1 -o /Users/jiuwen/AA/powermetrics_gpu_ceiling.txt

用法：
    /opt/homebrew/bin/python3.14 tools/gpu_power_ceiling_probe.py --duration 90
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=90.0, help="压载时长秒数")
    p.add_argument("--size", type=int, default=8192, help="方阵尺寸")
    args = p.parse_args()

    size = max(1024, args.size)
    dt = 0.0
    a = mx.random.normal((size, size))
    b = mx.random.normal((size, size))
    c = mx.random.normal((size, size))
    start = time.monotonic()
    it = 0
    try:
        while time.monotonic() - start < args.duration:
            c = (a @ b) + c
            c = c * 0.999
            a = a + (b * 0.00001)
            mx.eval(c, a)
            it += 1
            if it % 5 == 0:
                print(f"[probe] iter={it} elapsed={time.monotonic()-start:.1f}s", flush=True)
    finally:
        mx.eval(c, a)
        print(f"[probe] done iterations={it} seconds={time.monotonic()-start:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
