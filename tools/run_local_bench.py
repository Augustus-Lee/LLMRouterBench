# -*- coding: utf-8 -*-
"""从 config/local_model_eval.yaml 启动 bench_multi_query。

用法：
    /Users/augustus/miniconda3/bin/python tools/run_local_bench.py \
        --model-name qwen3.8-27b --dry-run

    /Users/augustus/miniconda3/bin/python tools/run_local_bench.py \
        --model-name qwen3.8-27b --limit-per-dataset 0

前置：对应推理服务已经启动并通过 smoke。
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

try:
    import bench_multi_query
except ImportError:  # python tools/run_local_bench.py 直跑
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench_multi_query  # type: ignore


def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("missing PyYAML; use /Users/augustus/miniconda3/bin/python") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _csv(values: Any) -> str:
    if isinstance(values, (list, tuple)):
        return ",".join(str(v) for v in values)
    return str(values)


def build_argv(model: Dict[str, Any], defaults: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    argv = ["--model", str(model["api_model_name"])]
    for key in ("engine", "engine_version", "weight_format", "quantization",
                "draft_model", "kv_format", "state_format", "model_revision_or_digest"):
        if model.get(key) not in (None, ""):
            argv += [f"--{key.replace('_', '-')}", str(model[key])]
    argv += ["--host", str(model.get("host", "127.0.0.1"))]
    argv += ["--port", str(int(model["port"]))]
    argv += ["--datasets", _csv(defaults.get("datasets", bench_multi_query.DEFAULT_DATASETS))]
    argv += ["--limit-per-dataset", str(args.limit_per_dataset if args.limit_per_dataset is not None else defaults.get("limit_per_dataset", 0))]
    argv += ["--concurrency", _csv(args.concurrency if args.concurrency else defaults.get("concurrency", [1, 2, 4, 8, 16]))]
    argv += ["--repeat", str(defaults.get("repeat", 2))]
    argv += ["--warmup", str(defaults.get("warmup", 1))]
    argv += ["--mode", str(defaults.get("mode", "both"))]
    argv += ["--max-output-tokens", str(defaults.get("max_output_tokens", 256))]
    argv += ["--temperature", str(defaults.get("temperature", 0.0))]
    argv += ["--top-p", str(defaults.get("top_p", 1.0))]
    argv += ["--seed", str(defaults.get("seed", 42))]
    argv += ["--max-attempts", str(defaults.get("max_attempts", 3))]
    argv += ["--backoff-base-s", str(defaults.get("backoff_base_s", 1.0))]
    if defaults.get("per_request_machine_metrics", False):
        argv += ["--per-request-machine-metrics"]
    if args.out:
        argv += ["--out", args.out]
    if args.dry_run:
        argv += ["--dry-run"]
    return argv


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="run_local_bench")
    parser.add_argument("--config", default="config/local_model_eval.yaml")
    parser.add_argument("--model-name", required=True,
                        choices=["qwen3.8-27b", "qwen3.6-35b-a3b", "gemma-4-26b-a4b"])
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--concurrency", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.exists(args.config):
        print(f"[run_local_bench] config not found: {args.config}")
        return 1
    config = load_config(args.config)
    defaults = config.get("run_defaults", {})
    model = next((m for m in config.get("models", []) if m["name"] == args.model_name), None)
    if model is None:
        print(f"[run_local_bench] model {args.model_name} not found in {args.config}")
        return 1

    bench_argv = build_argv(model, defaults, args)
    print(f"[run_local_bench] model={model['name']} "
          f"engine={model.get('engine')} port={model.get('port')}")
    return bench_multi_query.main(bench_argv)


if __name__ == "__main__":
    raise SystemExit(main())
