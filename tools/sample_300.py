# -*- coding: utf-8 -*-
"""固定 seed 随机抽样 300 题评测清单，供 bench_accuracy_concurrency 的 manifest 模式使用。

口径：6 个数据集各 50 题 = 300 题；与 sample_882.py 使用相同的确定性抽样规则。

产出：
    results/sample_300_seed42.jsonl，每行 {"dataset_id": "...", "record_index": N}
    record_index = 对应数据文件中的行号（0 起，不含空行）。

可复现性：同一 --seed 输出完全一致；打印清单 sha256。

用法：
    python tools/sample_300.py --seed 42 --out results/sample_300_seed42.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

# dataset_id -> (相对数据路径, 抽样本数)
DATASET_SPECS: Dict[str, Tuple[str, int]] = {
    "gpqa":    ("data/GPQA/gpqa_diamond.json", 50),
    "mmlupro": ("data/MMLUPro/test.json", 50),
    "math500": ("data/MATH500/test.json", 50),
    "arcc":    ("data/arc_c/arc_test.json", 50),
    "gsm8k":   ("data/GSM8K/test.json", 50),
    "mmlu":    ("data/MMLU/test.json", 50),
}
DS_ORDER = ["gpqa", "mmlupro", "math500", "arcc", "gsm8k", "mmlu"]


def sample_indices(population_size: int, sample_size: int, seed: int = 42) -> List[int]:
    """升序、无重复的抽样下标；sample_size 超过总体时取全量；确定性。"""
    if population_size <= 0 or sample_size <= 0:
        return []
    k = min(sample_size, population_size)
    if k == population_size:
        return list(range(population_size))
    return sorted(random.Random(seed).sample(range(population_size), k))


def count_jsonl_lines(path: Path) -> int:
    """统计非空行数（数据文件为逐行 JSON，扩展名 .json）。"""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def build_manifest(seed: int = 42) -> Tuple[List[Dict[str, int]], Dict[str, Dict[str, int]]]:
    """生成 300 题清单 + 每集抽样统计。"""
    entries: List[Dict[str, int]] = []
    stats: Dict[str, Dict[str, int]] = {}
    for ds in DS_ORDER:
        rel, target = DATASET_SPECS[ds]
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"数据文件缺失：{path}（检查 Git LFS 是否已拉取）")
        total = count_jsonl_lines(path)
        idxs = sample_indices(total, target, seed)
        for i in idxs:
            entries.append({"dataset_id": ds, "record_index": i})
        stats[ds] = {"total": total, "sampled": len(idxs)}
    return entries, stats


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sample_300",
        description="固定 seed 随机抽样 300 题评测清单（每数据集 50 题，manifest JSONL）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/sample_300_seed42.jsonl")
    args = parser.parse_args(argv)

    entries, stats = build_manifest(args.seed)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[sample] seed={args.seed} 共 {len(entries)} 题 -> {out}")
    total_sampled = 0
    for ds in DS_ORDER:
        s = stats[ds]
        total_sampled += s["sampled"]
        print(f"  {ds:8s} total={s['total']:5d} sampled={s['sampled']:4d}")
    print(f"[sample] 合计 sampled={total_sampled}")
    print(f"[sample] sha256={_sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
