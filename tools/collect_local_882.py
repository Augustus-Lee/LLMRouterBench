# -*- coding: utf-8 -*-
"""定向采集本地端侧模型答案矩阵：与 DeepSeek-V4.1-Flash 的 882 题严格对齐。

背景（同 collect_9b_882.py）：results/bench 里 DeepSeek 的 882 题是上一轮抽样冻结的矩阵，
data/ 现已是全量数据集（gpqa 198、mmlupro 1001、math500 500、arcc 1172 等），
直接跑 collector 会采大全集且 (dataset, record_index) 对齐不上。

本脚本逐题读 DeepSeek 记录（prompt/ground_truth/index），让本地模型
（Splash 18080/18081、MLX-VLM 18091 等 OpenAI 兼容服务）答同一题，
按原判分器打分，产出 results/bench/<ds>/test/<target>/*.json，
与 DeepSeek 完全同 (dataset, index) 对齐。

本地推理与云端的核心差异：
  - base_url 指向本地 OpenAI 兼容服务（无需真实 api_key，占位值 "ollama"）
  - 单并发（Splash --queue-size 1 / MLX-VLM --max-num-seqs 1 已固定）
  - timeout 默认 1800s（本地推理慢）
  - pricing 全 0（端侧成本=0 口径）

用法：
  python tools/collect_local_882.py --engine splash-qwen38 --limit 2    # 冒烟
  python tools/collect_local_882.py --engine splash-qwen38
  python tools/collect_local_882.py --engine splash-qwen36
  python tools/collect_local_882.py --engine mlxvlm-gemma
  python tools/collect_local_882.py --engine custom \
      --base-url http://127.0.0.1:18081/v1 \
      --api-model mlx-community/Qwen3.6-35B-A3B-4bit \
      --target-model Qwen3.6-35B-A3B

前置条件（缺一不可，否则脚本会显式报错）：
  - 3.5 上已有 results/bench/<ds>/test/DeepSeek-V4.1-Flash/*.json（882 题来源）
  - 目标本地服务已就绪（Splash /ready、MLX-VLM /health model_ready=true）
  - 已 pip install -r requirements.txt（openai/httpx/loguru/datasets 等重依赖）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_MODEL = "DeepSeek-V4.1-Flash"
DS_ORDER = ["gpqa", "mmlupro", "math500", "arcc", "gsm8k", "mmlu"]

# 本地引擎预设：端口与模型 ID 来自评测方案 v4（表 A/B/C + 表 D 端口 18082）
ENGINE_PRESETS: Dict[str, Dict[str, Any]] = {
    "splash-qwen38": {
        "target_model": "Qwen3.8-27B",
        "api_model": "Qwen/Qwen3.8-27B",
        "host": "127.0.0.1",
        "port": 18080,
    },
    "splash-qwen36": {
        "target_model": "Qwen3.6-35B-A3B",
        "api_model": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "host": "127.0.0.1",
        "port": 18081,
    },
    "mlxvlm-gemma": {
        "target_model": "Gemma-4-26B-A4B",
        "api_model": "/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit",
        "host": "127.0.0.1",
        "port": 18091,
    },
}


def build_base_url(host: str, port: int) -> str:
    """构造 OpenAI 兼容 base_url（生成器需要 /v1 前缀）。"""
    return f"http://{host}:{port}/v1"


def build_generator_config(
    target_model: str,
    api_model: str,
    base_url: str,
    api_key: str = "ollama",
    timeout: int = 1800,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造 generators.factory.create_generator 需要的配置字典。

    本地口径：端侧成本=0（pricing 全 0）、api_key 占位值、direct 生成器。
    """
    return {
        "name": target_model,
        "api_model_name": api_model,
        "base_url": base_url,
        "api_key": api_key,  # 本地占位值，不写真实云凭据
        "generator_type": "direct",
        "temperature": 0.0,
        "top_p": 1.0,
        "timeout": timeout,
        "extra_body": extra_body or {},
        "pricing": {
            "prompt_price_per_million": 0.0,
            "completion_price_per_million": 0.0,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_local_882",
        description="采集本地端侧模型答案矩阵（与 DeepSeek 882 题对齐）")
    parser.add_argument("--engine", required=True,
                        choices=list(ENGINE_PRESETS) + ["custom"],
                        help="本地引擎预设；custom 需显式给 --base-url/--api-model/--target-model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--api-model", default=None, help="API 模型 ID，覆盖预设")
    parser.add_argument("--target-model", default=None, help="结果目录模型名，覆盖预设")
    parser.add_argument("--base-url", default=None,
                        help="完整 base_url 覆盖（含 /v1），优先级最高")
    parser.add_argument("--api-key", default="ollama",
                        help="本地服务占位密钥（不写真实云凭据）")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="本地 queue-size 1，默认单并发")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="单请求超时秒数（本地推理慢）")
    parser.add_argument("--extra-body-json", default="{}",
                        help="透传 extra_body JSON，如 '{\"enable_thinking\": false}'")
    parser.add_argument("--limit", type=int, default=0,
                        help="每数据集最多采几题（0=全部），用于冒烟")
    parser.add_argument("--datasets", default="", help="逗号分隔，只采这些数据集")
    return parser


def resolve_target_config(args: argparse.Namespace) -> Dict[str, str]:
    """把 CLI 参数 + 预设解析为最终 (target_model, api_model, base_url)。"""
    preset = ENGINE_PRESETS.get(args.engine)
    if preset is not None:
        target_model = args.target_model or preset["target_model"]
        api_model = args.api_model or preset["api_model"]
        host = args.host
        port = args.port or preset["port"]
    else:  # custom
        target_model = args.target_model
        api_model = args.api_model
        host = args.host
        port = args.port

    if not target_model or not api_model or not port:
        raise SystemExit(
            "信息不足：--engine custom 时必须提供 --api-model、--target-model、--port"
            "（或改用预设 --engine splash-qwen38/splash-qwen36/mlxvlm-gemma）")

    base_url = args.base_url or build_base_url(host, port)
    return {
        "target_model": target_model,
        "api_model": api_model,
        "base_url": base_url,
    }


def _load_smoke_cleanup(limit: int, target_model: str) -> None:
    """--limit 冒烟产生的残缺结果要删除，避免与正式全量文件混淆（同 collect_9b_882）。"""
    if not limit:
        return
    base = ROOT / "results" / "bench"
    for ds in DS_ORDER:
        d = base / ds / "test" / target_model
        if not d.exists():
            continue
        for f in d.glob(f"{ds}-test-{target_model}-*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data and data.get("counts") == limit:
                f.unlink()
                print(f"  [rm smoke] {f.name}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    target = resolve_target_config(args)
    try:
        extra_body = json.loads(args.extra_body_json)
    except ValueError as exc:
        raise SystemExit(f"--extra-body-json 不是合法 JSON: {exc}") from exc

    # 重依赖懒加载：本机无 openai/httpx/datasets 时仍可 --help/纯逻辑测试；
    # 正式采集在 3.5 上装好 requirements 后运行才触发导入。
    from baselines.data_loader import BaselineDataLoader
    from evaluation.factory import EvaluatorFactory
    from generators.factory import create_generator
    from data_collector.storage import (  # noqa: F401
        ResultsStorage, BenchmarkResult, RecordResult)
    from data_collector.storage import ResultsStorage as _RS  # 冒烟清理用

    loader = BaselineDataLoader(
        config_path=str(ROOT / "config" / "baseline_config_edge.yaml"))
    records = [r for r in loader.load_all_records() if r.model_name == SOURCE_MODEL]
    if not records:
        raise SystemExit(
            f"未找到 {SOURCE_MODEL} 的 882 题来源记录："
            f"请确认 results/bench/<ds>/test/{SOURCE_MODEL}/ 存在且 config/"
            f"baseline_config_edge.yaml 指向正确")

    by_ds: Dict[str, List[Any]] = {}
    for r in records:
        by_ds.setdefault(r.dataset_id, []).append(r)
    for ds in by_ds:
        by_ds[ds].sort(key=lambda r: r.record_index)
    print(f"[load] {len(records)} 题 / {len(by_ds)} 数据集（来源 {SOURCE_MODEL}）")
    print(f"[target] {target['target_model']} @ {target['base_url']}")

    only = {d.strip() for d in args.datasets.split(",") if d.strip()}
    generator = create_generator(build_generator_config(
        target_model=target["target_model"],
        api_model=target["api_model"],
        base_url=target["base_url"],
        api_key=args.api_key,
        timeout=args.timeout,
        extra_body=extra_body,
    ), cache_config=None)
    evaluator_factory = EvaluatorFactory()
    storage = ResultsStorage(base_dir=str(ROOT / "results"))

    for ds in DS_ORDER:
        if ds not in by_ds or (only and ds not in only):
            continue
        recs = by_ds[ds]
        if args.limit:
            recs = recs[: args.limit]
        evaluator = evaluator_factory.get_evaluator(ds)
        print(f"[run] {ds}: {len(recs)} 题", flush=True)

        def process(rec):
            gen = generator.generate(rec.prompt)
            ev = evaluator.evaluate({"answer": rec.ground_truth}, gen.output)
            is_correct = ev.get("is_correct", False)
            if is_correct is None:
                score = None
            elif isinstance(is_correct, bool):
                score = 1.0 if is_correct else 0.0
            else:
                score = float(is_correct)
            extra = {k: v for k, v in ev.items()
                     if k not in ("is_correct", "prediction", "ground_truth")}
            return RecordResult(
                index=rec.record_index,          # 与 DeepSeek 同 index，保证对齐
                origin_query=rec.origin_query,
                prompt=rec.prompt,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                cost=gen.cost,
                score=score,
                prediction=ev.get("prediction", ""),
                ground_truth=ev.get("ground_truth", rec.ground_truth),
                raw_output=gen.output,
                extra_fields=extra,
            )

        results = []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(process, r) for r in recs]
            done = 0
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:  # 单题失败不中断整个数据集
                    print(f"  [error] {exc}", flush=True)
                done += 1
                if done % 50 == 0 or done == len(recs):
                    print(f"  [{ds}] {done}/{len(recs)}", flush=True)

        results.sort(key=lambda r: r.index)
        total_pt = sum(r.prompt_tokens or 0 for r in results)
        total_ct = sum(r.completion_tokens or 0 for r in results)
        total_cost = sum(r.cost or 0.0 for r in results)
        valid = [r.score for r in results if r.score is not None]
        perf = sum(valid) / len(valid) if valid else 0.0

        result = BenchmarkResult(
            performance=perf, time_taken=0.0,
            prompt_tokens=total_pt, completion_tokens=total_ct, cost=total_cost,
            counts=len(results), model_name=target["target_model"],
            dataset_name=ds, split="test",
            records=results, demo=False, extra_metrics={},
        )
        storage.save_result(result, ds, "test", target["target_model"], data_fingerprint="")
        print(f"  [saved] {ds}: acc={perf * 100:.2f}%  cost={total_cost:.4f}  "
              f"n={len(results)}", flush=True)

    _load_smoke_cleanup(args.limit, target["target_model"])
    print("[all done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
