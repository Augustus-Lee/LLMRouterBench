# -*- coding: utf-8 -*-
"""Splash 1.5.0 性能评测适配器（表 A / 表 B）。

用法示例（Qwen3.6-35B-A3B，方案 5.3/5.7 节）：

    python tools/bench_splash.py \\
        --model 'mlx-community/Qwen3.6-35B-A3B-4bit' \\
        --port 18081 \\
        --draft-model 'Qwen3.6-35B-A3B-DFlash2' \\
        --weight-format mlx --quantization 4bit-affine \\
        --kv-format int8 --state-format bf16 \\
        --cold-load-s 41.3 \\
        --server-pid $(cat ~/llm-deploy/run/splash-qwen36.pid) \\
        --monitor-disk disk0

前置条件：目标 Splash 服务已在 127.0.0.1:18081 就绪（/ready 通过）。
冷加载时间由启动方测量（进程启动 -> /ready），通过 --cold-load-s 传入。
"""
from __future__ import annotations

import argparse
from typing import List, Optional

try:  # 支持包导入与脚本直跑两种方式
    from tools.bench_common import (  # noqa: F401
        DEFAULT_CONTEXT_LADDER, OpenAICompatClient, run_benchmark)
except ImportError:  # python tools/bench_splash.py 直跑
    from bench_common import (  # type: ignore # noqa: F401
        DEFAULT_CONTEXT_LADDER, OpenAICompatClient, run_benchmark)


def _parse_contexts(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_splash",
        description="Splash 引擎文本性能评测（上下文阶梯矩阵）")
    parser.add_argument("--model", required=True,
                        help="API 显示 ID（必填，本地加载也必须提供）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081,
                        help="Qwen3.8=18080 / Qwen3.6=18081（默认 18081）")
    parser.add_argument("--engine-version", default="1.5.0")
    parser.add_argument("--draft-model", default=None,
                        help="DFlash2 draft 名称；不能跨模型复用")
    parser.add_argument("--weight-format", default=None,
                        help="gguf 或 mlx（必须标注，不混表）")
    parser.add_argument("--quantization", default=None,
                        help="如 Q4_K_M / 4bit-affine")
    parser.add_argument("--kv-format", default="int8",
                        help="Splash 全程固定 int8（方案 5.7 节）")
    parser.add_argument("--state-format", default="bf16")
    parser.add_argument("--revision-or-digest", default=None)
    parser.add_argument("--contexts", type=_parse_contexts,
                        default=list(DEFAULT_CONTEXT_LADDER),
                        help="逗号分隔上下文阶梯，默认 1K/8K/32K/64K/128K")
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cold-load-s", type=float, default=None,
                        help="进程启动到 /ready 的秒数（启动方测量后传入）")
    parser.add_argument("--server-pid", type=int, default=None,
                        help="Splash 进程 PID，用于 psutil RSS 采样（可选）")
    parser.add_argument("--monitor-disk", default=None,
                        help="iostat 磁盘名（如 disk0），检测 SSD 流式读取（可选）")
    parser.add_argument("--out", default=None,
                        help="输出目录，默认 results/mac_benchmark/raw/splash_<modeltag>")
    parser.set_defaults(engine="splash")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.engine = "splash"
    if args.out is None:
        tag = args.model.replace("/", "_").replace(":", "_")
        args.out = f"results/mac_benchmark/raw/splash_{tag}"
    client = OpenAICompatClient(host=args.host, port=args.port, model=args.model)
    paths = run_benchmark(args, client)
    print(f"[bench_splash] done, {len(paths)} record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
