# -*- coding: utf-8 -*-
"""MLX-VLM 0.8.0 性能评测适配器（表 C，Gemma-4-26B-A4B 专用）。

用法示例（方案 5.4/5.7 节）：

    python tools/bench_mlxvlm.py \\
        --model /Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit \\
        --port 18091 \\
        --weight-format mlx --quantization 4bit-affine \\
        --cold-load-s 38.0 \\
        --server-pid $(cat ~/llm-deploy/run/gemma4-server.pid) \\
        --monitor-disk disk0

前置条件：MLX-VLM 服务已就绪（/health 返回 model_ready=true），
且以 --max-num-seqs 1 启动，保证单请求串行口径。
结果不得与 Splash 表混排（跨引擎只做观察，不做排名）。
"""
from __future__ import annotations

import argparse
from typing import List, Optional

try:  # 支持包导入与脚本直跑两种方式
    from tools.bench_common import (  # noqa: F401
        DEFAULT_CONTEXT_LADDER, OpenAICompatClient, run_benchmark)
except ImportError:  # python tools/bench_mlxvlm.py 直跑
    from bench_common import (  # type: ignore # noqa: F401
        DEFAULT_CONTEXT_LADDER, OpenAICompatClient, run_benchmark)


def _parse_contexts(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_mlxvlm",
        description="MLX-VLM 引擎文本性能评测（上下文阶梯矩阵）")
    parser.add_argument("--model", required=True,
                        help="模型目录路径（API 显示 ID）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--engine-version", default="0.8.0")
    parser.add_argument("--weight-format", default="mlx")
    parser.add_argument("--quantization", default="4bit-affine")
    parser.add_argument("--revision-or-digest", default=None)
    parser.add_argument("--contexts", type=_parse_contexts,
                        default=list(DEFAULT_CONTEXT_LADDER))
    parser.add_argument("--workload-file", default=None,
                        help="由 tools/build_perf_workload.py 生成的模型专属 JSONL；"
                             "提供后按 target_tokens 与 --contexts 过滤并逐条执行")
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cold-load-s", type=float, default=None)
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--monitor-disk", default=None)
    parser.add_argument("--out", default=None)
    parser.set_defaults(engine="mlx-vlm")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.engine = "mlx-vlm"
    # MLX-VLM 无 draft / KV 精度参数；保证 schema 字段存在且为 None
    args.draft_model = None
    args.kv_format = None
    args.state_format = None
    if args.out is None:
        tag = args.model.replace("/", "_").rstrip("/").split("/")[-1]
        args.out = f"results/mac_benchmark/raw/mlxvlm_{tag}"
    client = OpenAICompatClient(host=args.host, port=args.port, model=args.model)
    paths = run_benchmark(args, client)
    print(f"[bench_mlxvlm] done, {len(paths)} record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
