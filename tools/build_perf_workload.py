#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mac 文本模型性能 workload 构建器。

设计目标
--------
1. 生成“受控长度、真实语料、长上下文”三层性能 workload；
2. 不依赖 torch/transformers/datasets，默认使用 Hugging Face
   datasets-server 行接口和 tokenizers 做长度校准；
3. 所有公共来源固定 revision、license、split 和样本 ID；
4. 不把 MMLU/GSM8K 等能力题答案写入性能集。

典型用法
--------
    python tools/build_perf_workload.py download --source all
    python tools/build_perf_workload.py build
    python tools/build_perf_workload.py audit

生成目录：``benchmark_data/performance/``。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
PERF_ROOT = ROOT / "benchmark_data" / "performance"
SOURCES_DIR = PERF_ROOT / "sources"
GENERATED_DIR = PERF_ROOT / "generated"
TOKENIZERS_DIR = PERF_ROOT / "tokenizers"
SPECS_DIR = PERF_ROOT / "specs"

SCHEMA_VERSION = "perf-workload-v1"
SEED = 42
CONTEXTS = (1024, 8192, 32768, 65536, 131072, 262144)
REFERENCE_MODEL = "qwen3.6-35b-a3b"

# 只放公开、可追溯且与性能输入无关乎标准答案的文本来源。
SOURCE_SPECS: Dict[str, Dict[str, Any]] = {
    "dolly": {
        "dataset": "databricks/databricks-dolly-15k",
        "short_name": "dolly",
        "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "license": "CC-BY-SA-3.0",
        "split": "train",
        "config": "default",
        "row_count": 100,
        "kind": "rows",
    },
    "ultrachat": {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "short_name": "ultrachat",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "license": "MIT",
        "split": "test_sft",
        "config": "default",
        "row_count": 100,
        "kind": "rows",
    },
    "longbench": {
        "dataset": "zai-org/LongBench-v2",
        "short_name": "longbench-v2",
        "revision": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
        "license": "Apache-2.0",
        "split": "train",
        "config": "default",
        "row_count": 3,
        "kind": "rows",
    },
}

TOKENIZER_SPECS: Dict[str, Dict[str, Any]] = {
    "qwen3.8-27b": {
        "model": "Qwen/Qwen3.8-27B",
        "revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
        "license": "Apache-2.0",
    },
    "qwen3.6-35b-a3b": {
        "model": "Qwen/Qwen3.6-35B-A3B",
        "revision": "345d467b117d1fd5eda9b879a5988de9ae5f47a7",
        "license": "Apache-2.0",
    },
    "gemma-4-26b-a4b": {
        "model": "google/gemma-4-26b-a4b-it",
        "revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        "license": "Apache-2.0",
    },
}

# 受控层每档样本数；256K 永远单独标记为 boundary_probe。
CONTROLLED_COUNTS = {
    1024: 3,
    8192: 3,
    32768: 3,
    65536: 3,
    131072: 3,
    262144: 1,
}
REAL_COUNTS = {
    "dolly": {1024: 2, 8192: 2},
    "ultrachat": {1024: 2, 8192: 2, 32768: 1},
}
LONG_COUNTS = {32768: 2, 65536: 2, 131072: 2, 262144: 1}


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            records.append(record)
    return records


def _request(url: str, timeout: float = 180.0, retries: int = 4) -> bytes:
    headers = {"User-Agent": "LLMRouterBench-perf-workload/1.0"}
    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_exc = exc
            if attempt < retries:
                wait = min(10, 2 ** attempt)
                print(f"[retry] {attempt}/{retries} {url}: {exc} (sleep {wait}s)",
                      file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_exc}")


def fetch_json(url: str, timeout: float = 180.0, retries: int = 4) -> Dict[str, Any]:
    body = _request(url, timeout=timeout, retries=retries)
    try:
        return json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"invalid JSON from {url}: {exc}") from exc


def rows_url(spec: Dict[str, Any]) -> str:
    query = urllib.parse.urlencode({
        "dataset": spec["dataset"],
        "config": spec["config"],
        "split": spec["split"],
        "revision": spec["revision"],
        "offset": 0,
        "length": spec["row_count"],
    })
    return "https://datasets-server.huggingface.co/rows?" + query


def normalize_source_rows(source_name: str,
                          payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 datasets-server 的 rows 响应转换为稳定的源记录。"""
    out: List[Dict[str, Any]] = []
    for item in payload.get("rows", []):
        row_idx = item.get("row_idx")
        row = item.get("row") or {}
        if source_name == "dolly":
            text = "Instruction: " + str(row.get("instruction") or "")
            context = str(row.get("context") or "")
            if context:
                text += "\n\nContext:\n" + context
            source_id = str(row_idx)
            extra = {"category": row.get("category")}
        elif source_name == "ultrachat":
            messages = row.get("messages") or []
            lines: List[str] = []
            for msg in messages:
                role = str(msg.get("role") or "unknown")
                content = str(msg.get("content") or "")
                lines.append(f"{role.capitalize()}: {content}")
            text = "\n\n".join(lines)
            source_id = str(row.get("prompt_id") or row_idx)
            extra = {"prompt_id": row.get("prompt_id"), "message_count": len(messages)}
        elif source_name == "longbench":
            context = str(row.get("context") or "")
            question = str(row.get("question") or "")
            text = context
            source_id = str(row.get("_id") or row_idx)
            extra = {
                "question": question,
                "domain": row.get("domain"),
                "sub_domain": row.get("sub_domain"),
                "length": row.get("length"),
                "difficulty": row.get("difficulty"),
            }
        else:
            raise ValueError(f"unsupported source: {source_name}")

        if not text.strip():
            continue
        out.append({
            "source_dataset": SOURCE_SPECS[source_name]["dataset"],
            "source_revision": SOURCE_SPECS[source_name]["revision"],
            "source_split": SOURCE_SPECS[source_name]["split"],
            "source_id": source_id,
            "source_row_idx": row_idx,
            "source_text": text,
            "source_text_sha256": sha256_text(text),
            **extra,
        })
    return out


def download_source(source_name: str, force: bool = False) -> Dict[str, Any]:
    if source_name not in SOURCE_SPECS:
        raise ValueError(f"unknown source: {source_name}")
    spec = SOURCE_SPECS[source_name]
    out_dir = SOURCES_DIR / spec["short_name"]
    rows_path = out_dir / "rows.json"
    jsonl_path = out_dir / "records.jsonl"
    meta_path = out_dir / "metadata.json"
    if jsonl_path.exists() and meta_path.exists() and not force:
        print(f"[skip] {source_name}: existing {jsonl_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    url = rows_url(spec)
    print(f"[download] {source_name}: {url}")
    payload = fetch_json(url)
    records = normalize_source_rows(source_name, payload)
    if not records:
        raise RuntimeError(f"{source_name}: no non-empty records returned")
    write_json(rows_path, payload)
    write_jsonl(jsonl_path, records)
    meta = {
        "source": source_name,
        "dataset": spec["dataset"],
        "revision": spec["revision"],
        "license": spec["license"],
        "split": spec["split"],
        "config": spec["config"],
        "endpoint": url,
        "fetched_at": _now_utc(),
        "num_rows_returned": len(payload.get("rows", [])),
        "num_records": len(records),
        "rows_sha256": sha256_file(rows_path),
        "records_sha256": sha256_file(jsonl_path),
    }
    write_json(meta_path, meta)
    print(f"[ok] {source_name}: {len(records)} records -> {jsonl_path}")
    return meta


def download_tokenizer(model_slug: str, force: bool = False) -> Dict[str, Any]:
    if model_slug not in TOKENIZER_SPECS:
        raise ValueError(f"unknown tokenizer: {model_slug}")
    spec = TOKENIZER_SPECS[model_slug]
    out_dir = TOKENIZERS_DIR / model_slug
    path = out_dir / "tokenizer.json"
    meta_path = out_dir / "metadata.json"
    if path.exists() and meta_path.exists() and not force:
        print(f"[skip] tokenizer {model_slug}: existing {path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    url = (f"https://huggingface.co/{spec['model']}/resolve/"
           f"{spec['revision']}/tokenizer.json")
    print(f"[download] tokenizer {model_slug}: {url}")
    body = _request(url, timeout=240.0, retries=5)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes(body)
    tmp.replace(path)
    meta = {
        "model_slug": model_slug,
        "model": spec["model"],
        "model_revision": spec["revision"],
        "license": spec["license"],
        "url": url,
        "fetched_at": _now_utc(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    write_json(meta_path, meta)
    print(f"[ok] tokenizer {model_slug}: {path.stat().st_size} bytes")
    return meta


def load_source_records(source_name: str) -> List[Dict[str, Any]]:
    spec = SOURCE_SPECS[source_name]
    path = SOURCES_DIR / spec["short_name"] / "records.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run: python tools/build_perf_workload.py download --source {source_name}")
    return read_jsonl(path)


class TokenCounter:
    """tokenizers.Tokenizer 的可选薄封装；缺失时明确标记为估算。"""

    def __init__(self, tokenizer_path: Optional[Path] = None) -> None:
        self.path = tokenizer_path
        self._tokenizer = None
        if tokenizer_path is not None:
            try:
                from tokenizers import Tokenizer  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "tokenizers is not installed; use the miniconda Python or install tokenizers") from exc
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    @property
    def exact(self) -> bool:
        return self._tokenizer is not None

    def count(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
        return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """无 tokenizer 时的保守粗略估计；正式结果必须标记 exact=false。"""
    if not text:
        return 0
    # 英文/符号按空白词，CJK 按字；仅作为构建阶段兜底。
    cjk = re.findall(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", text)
    non_cjk = re.sub(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", " ", text)
    return len(cjk) + len(non_cjk.split())


def _repeat_to_min(text: str, min_chars: int, variant: int) -> str:
    if not text:
        return text
    marker = f"\n\n--- variant {variant} ---\n\n"
    out = text
    while len(out) < max(min_chars, 1):
        out += marker + text
    return out


def _fit_prompt(prompt_builder, body: str, target_tokens: int,
                counter: TokenCounter, variant: int) -> Tuple[str, int]:
    """在 token 预算内截断正文，返回完整 prompt 和实际 token 数。"""
    # 先确保正文至少达到目标的粗略字符量，避免 256K 样本因源文本不足失败。
    body = _repeat_to_min(body, target_tokens * 3, variant)
    full = prompt_builder(body)
    if counter.count(full) < target_tokens:
        body = _repeat_to_min(body, target_tokens * 6, variant)
        full = prompt_builder(body)

    # 字符级二分通常足够稳定；token 边界只造成极小的非单调。
    lo, hi = 0, len(body)
    best_body = body
    best_count = counter.count(full)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = body[:mid].rstrip()
        candidate_prompt = prompt_builder(candidate)
        count = counter.count(candidate_prompt)
        if count <= target_tokens:
            if count > best_count or best_body == body:
                best_body, best_count = candidate, count
            lo = mid + 1
        else:
            hi = mid - 1
    prompt = prompt_builder(best_body)
    return prompt, counter.count(prompt)


def _controlled_builder(body: str) -> str:
    return (
        "Read the following reference document and provide a concise summary. "
        "This is a performance workload; no factual answer is required.\n\n"
        "[REFERENCE DOCUMENT]\n" + body +
        "\n\n[INSTRUCTION]\nSummarize the document in no more than 256 tokens."
    )


def _real_builder(body: str) -> str:
    return (
        "You are a helpful assistant. Continue the request below with a short answer.\n\n"
        "[REQUEST]\n" + body
    )


def _long_builder(question: str):
    def build(body: str) -> str:
        return (
            "Read the document and answer the question. This workload measures long-context "
            "inference performance; answer briefly.\n\n"
            "[DOCUMENT]\n" + body +
            "\n\n[QUESTION]\n" + question
        )
    return build


def _select_source_body(source_records: Sequence[Dict[str, Any]],
                        variant: int, min_chars: int) -> str:
    if not source_records:
        raise ValueError("empty source records")
    # 轮换起点，避免所有样本都以同一段开头。
    ordered = list(source_records[variant % len(source_records):]) + \
              list(source_records[:variant % len(source_records)])
    body = "\n\n--- source boundary ---\n\n".join(
        str(r.get("source_text") or "") for r in ordered)
    return _repeat_to_min(body, min_chars, variant)


def _source_ids(records: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(r.get("source_id")) for r in records]


def _base_record(*, workload_id: str, layer: str, target: int, prompt: str,
                 actual: int, model_slug: str, counter: TokenCounter,
                 source: Dict[str, Any], construction: str, variant: int,
                 question: Optional[str] = None,
                 token_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    token_counts = token_counts or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "id": workload_id,
        "layer": layer,
        "boundary_probe": target >= 262144,
        "target_tokens": target,
        "actual_tokens": actual,
        "tokenizer": model_slug,
        "tokenizer_model": TOKENIZER_SPECS[model_slug]["model"],
        "tokenizer_revision": TOKENIZER_SPECS[model_slug]["revision"],
        "tokenizer_exact": counter.exact,
        "text": prompt,
        "text_sha256": sha256_text(prompt),
        "source_dataset": source.get("source_dataset"),
        "source_revision": source.get("source_revision"),
        "source_split": source.get("source_split"),
        "source_ids": source.get("source_ids", []),
        "source_license": source.get("source_license"),
        "construction": construction,
        "variant": variant,
        "seed": SEED,
        "question_included": question is not None,
        "token_counts": token_counts,
        "notes": "Performance workload only; not a quality/accuracy benchmark.",
    }


def _load_tokenizers() -> Dict[str, TokenCounter]:
    counters: Dict[str, TokenCounter] = {}
    for slug in TOKENIZER_SPECS:
        path = TOKENIZERS_DIR / slug / "tokenizer.json"
        if path.exists():
            try:
                counters[slug] = TokenCounter(path)
                print(f"[tokenizer] {slug}: exact, sha256={sha256_file(path)[:12]}")
            except Exception as exc:
                print(f"[warn] tokenizer {slug} unusable: {exc}", file=sys.stderr)
        else:
            print(f"[warn] tokenizer {slug} missing; using estimate", file=sys.stderr)
    if not counters:
        raise RuntimeError("no tokenizer available; run download --source tokenizers")
    return counters


def _make_source_meta(source_name: str, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    spec = SOURCE_SPECS[source_name]
    return {
        "source_dataset": spec["dataset"],
        "source_revision": spec["revision"],
        "source_split": spec["split"],
        "source_license": spec["license"],
        "source_ids": _source_ids(records),
    }


def build_workloads() -> Dict[str, Any]:
    """生成按模型校准的 JSONL 和三源合并 all.jsonl。"""
    dolly = load_source_records("dolly")
    ultrachat = load_source_records("ultrachat")
    longbench = load_source_records("longbench")
    counters = _load_tokenizers()

    model_records: Dict[str, List[Dict[str, Any]]] = {slug: [] for slug in counters}

    # 三个模型的 tokenizer 不同；每个模型生成自己的长度校准文件。
    for slug, counter in counters.items():
        records: List[Dict[str, Any]] = []

        # ---- Layer 1: controlled length ----
        for target, count in CONTROLLED_COUNTS.items():
            for variant in range(count):
                selected = [longbench[variant % len(longbench)]]
                body = _select_source_body(selected, variant, target * 3)
                prompt, actual = _fit_prompt(
                    _controlled_builder, body, target, counter, variant)
                wid = f"controlled-{target}-{variant + 1:02d}"
                records.append(_base_record(
                    workload_id=wid, layer="controlled", target=target,
                    prompt=prompt, actual=actual, model_slug=slug,
                    counter=counter, source=_make_source_meta("longbench", selected),
                    construction="prefix_fit_and_repeat_public_text", variant=variant,
                ))

        # ---- Layer 2: real requests ----
        for source_name, targets in REAL_COUNTS.items():
            rows = dolly if source_name == "dolly" else ultrachat
            for target, count in targets.items():
                for variant in range(count):
                    n_take = max(1, min(len(rows), 2 + variant))
                    selected = [rows[(variant + i) % len(rows)] for i in range(n_take)]
                    body = _select_source_body(selected, variant, target * 3)
                    prompt, actual = _fit_prompt(
                        _real_builder, body, target, counter, variant)
                    wid = f"real-{source_name}-{target}-{variant + 1:02d}"
                    records.append(_base_record(
                        workload_id=wid, layer="real", target=target,
                        prompt=prompt, actual=actual, model_slug=slug,
                        counter=counter, source=_make_source_meta(source_name, selected),
                        construction="public_request_concat_and_fit", variant=variant,
                    ))

        # ---- Layer 3: long context ----
        for target, count in LONG_COUNTS.items():
            for variant in range(count):
                selected = [longbench[variant % len(longbench)]]
                body = _select_source_body(selected, variant, target * 3)
                question = str(selected[0].get("question") or
                               "Summarize the document in one short paragraph.")
                builder = _long_builder(question)
                prompt, actual = _fit_prompt(builder, body, target, counter, variant)
                wid = f"long-longbench-{target}-{variant + 1:02d}"
                records.append(_base_record(
                    workload_id=wid, layer="long_context", target=target,
                    prompt=prompt, actual=actual, model_slug=slug,
                    counter=counter, source=_make_source_meta("longbench", selected),
                    construction="public_long_document_plus_question", variant=variant,
                    question=question,
                ))

        # 全 tokenizer 计数：给同一个 prompt 打上其他模型的 token 数，
        # 便于分析跨 tokenizer 的长度偏差。
        for record in records:
            counts = {}
            for other_slug, other in counters.items():
                counts[other_slug] = other.count(record["text"])
            record["token_counts"] = counts
        model_records[slug] = records

    # 参考文件使用 Qwen3.6 的校准结果，并保留全部模型计数。
    reference_records = model_records.get(REFERENCE_MODEL) or next(iter(model_records.values()))
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(GENERATED_DIR / "all.jsonl", reference_records)
    by_model_dir = GENERATED_DIR / "by_model"
    by_model_dir.mkdir(parents=True, exist_ok=True)
    for slug, records in model_records.items():
        write_jsonl(by_model_dir / f"{slug}.jsonl", records)
    for layer in ("controlled", "real", "long_context"):
        layer_records = [r for r in reference_records if r["layer"] == layer]
        write_jsonl(GENERATED_DIR / f"{layer}.jsonl", layer_records)

    # 规格文件固定保存生成参数，便于审计。
    write_json(SPECS_DIR / "public_sources.json", {
        "schema_version": SCHEMA_VERSION,
        "sources": SOURCE_SPECS,
        "tokenizers": TOKENIZER_SPECS,
        "controlled_counts": {str(k): v for k, v in CONTROLLED_COUNTS.items()},
        "real_counts": {k: {str(t): n for t, n in v.items()} for k, v in REAL_COUNTS.items()},
        "long_counts": {str(k): v for k, v in LONG_COUNTS.items()},
        "notes": [
            "No ground-truth answers are included.",
            "256K samples are boundary probes, not the default production matrix.",
            "Each model-specific file is tokenizer-calibrated; actual server prompt_tokens remains authoritative.",
        ],
    })

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_utc(),
        "generator": "tools/build_perf_workload.py",
        "seed": SEED,
        "reference_model": REFERENCE_MODEL,
        "context_ladder": list(CONTEXTS),
        "source_revisions": {
            name: {
                "dataset": spec["dataset"],
                "revision": spec["revision"],
                "license": spec["license"],
            } for name, spec in SOURCE_SPECS.items()
        },
        "tokenizers": {
            slug: {
                "model": spec["model"],
                "revision": spec["revision"],
                "license": spec["license"],
                "path": str(TOKENIZERS_DIR / slug / "tokenizer.json"),
            } for slug, spec in TOKENIZER_SPECS.items()
        },
        "files": {},
        "counts": {},
        "notes": [
            "all.jsonl uses the reference tokenizer (Qwen3.6) for length calibration.",
            "by_model/*.jsonl are independently tokenizer-calibrated.",
            "Use the server response usage.prompt_tokens as the authoritative measured token count.",
        ],
    }
    for slug, records in model_records.items():
        manifest["counts"][slug] = {
            "total": len(records),
            "by_layer": {
                layer: sum(1 for r in records if r["layer"] == layer)
                for layer in ("controlled", "real", "long_context")
            },
        }
    for path in sorted(GENERATED_DIR.rglob("*.jsonl")):
        manifest["files"][str(path.relative_to(PERF_ROOT))] = {
            "sha256": sha256_file(path),
            "records": len(read_jsonl(path)),
            "bytes": path.stat().st_size,
        }
    write_json(PERF_ROOT / "manifest.json", manifest)
    print(f"[ok] generated {sum(len(v) for v in model_records.values())} calibrated records")
    print(f"[ok] manifest: {PERF_ROOT / 'manifest.json'}")
    return manifest


def _audit_records(records: Sequence[Dict[str, Any]], label: str,
                   expected_tokenizer: Optional[str] = None
                   ) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    ids: set = set()
    required = (
        "id", "layer", "target_tokens", "actual_tokens", "tokenizer",
        "text", "text_sha256", "source_dataset", "source_revision",
        "source_license", "construction", "seed",
    )
    for i, rec in enumerate(records, 1):
        rid = rec.get("id")
        if not rid:
            errors.append(f"{label}: record {i}: missing id")
            continue
        if rid in ids:
            errors.append(f"{label}: duplicate id: {rid}")
        ids.add(rid)
        for key in required:
            if key not in rec or rec[key] in (None, ""):
                errors.append(f"{label}: {rid}: missing {key}")
        if expected_tokenizer and rec.get("tokenizer") != expected_tokenizer:
            errors.append(
                f"{label}: {rid}: tokenizer {rec.get('tokenizer')} != {expected_tokenizer}")
        text = rec.get("text") or ""
        if not text.strip():
            errors.append(f"{label}: {rid}: empty text")
        if sha256_text(text) != rec.get("text_sha256"):
            errors.append(f"{label}: {rid}: text_sha256 mismatch")
        target = rec.get("target_tokens")
        actual = rec.get("actual_tokens")
        if not isinstance(target, int) or not isinstance(actual, int):
            errors.append(f"{label}: {rid}: target/actual token is not int")
        else:
            # BPE 边界可能使结果略小；要求至少 97%，并且不能超过目标。
            if actual > target:
                errors.append(f"{label}: {rid}: actual {actual} > target {target}")
            elif actual < int(target * 0.97):
                errors.append(f"{label}: {rid}: actual {actual} < 97% of target {target}")
            if target >= 262144 and not rec.get("boundary_probe"):
                errors.append(f"{label}: {rid}: 256K sample not marked boundary_probe")
            if target < 262144 and rec.get("boundary_probe"):
                errors.append(
                    f"{label}: {rid}: non-256K sample incorrectly marked boundary_probe")
        for key in ("source_dataset", "source_revision", "source_license"):
            if rec.get(key) in (None, ""):
                errors.append(f"{label}: {rid}: missing provenance {key}")
        if rec.get("layer") == "long_context" and not rec.get("question_included"):
            warnings.append(f"{label}: {rid}: long_context sample has no question flag")
        counts = rec.get("token_counts")
        if not isinstance(counts, dict) or not counts:
            errors.append(f"{label}: {rid}: missing cross-tokenizer token_counts")
    return errors, warnings


def audit_workload(root: Path = PERF_ROOT) -> Dict[str, Any]:
    """检查所有生成文件、manifest 哈希、来源、256K 标记和 token 容差。"""
    all_path = root / "generated" / "all.jsonl"
    if not all_path.exists():
        raise FileNotFoundError(f"missing {all_path}; run build first")

    paths = [all_path]
    by_model_dir = root / "generated" / "by_model"
    if by_model_dir.exists():
        paths.extend(sorted(by_model_dir.glob("*.jsonl")))
    missing = [slug for slug in TOKENIZER_SPECS
               if not (by_model_dir / f"{slug}.jsonl").exists()]
    errors: List[str] = [f"missing model workload: {slug}" for slug in missing]
    warnings: List[str] = []
    datasets: Dict[str, Any] = {}
    for path in paths:
        records = read_jsonl(path)
        label = str(path.relative_to(root))
        expected = None
        if path.parent == by_model_dir:
            expected = path.stem
        file_errors, file_warnings = _audit_records(records, label, expected)
        errors.extend(file_errors)
        warnings.extend(file_warnings)
        datasets[label] = {
            "records": len(records),
            "unique_ids": len({r.get("id") for r in records if r.get("id")}),
            "sha256": sha256_file(path),
            "ok": not file_errors,
        }

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        errors.append("missing manifest.json")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for rel, meta in manifest.get("files", {}).items():
            path = root / rel
            if not path.exists():
                errors.append(f"manifest references missing file: {rel}")
            elif sha256_file(path) != meta.get("sha256"):
                errors.append(f"manifest sha256 mismatch: {rel}")
        for source_name, meta in manifest.get("source_revisions", {}).items():
            spec = SOURCE_SPECS.get(source_name)
            if not spec:
                errors.append(f"manifest has unknown source: {source_name}")
                continue
            for key in ("dataset", "revision", "license"):
                if meta.get(key) != spec.get(key):
                    errors.append(f"manifest source {source_name} {key} mismatch")

    report = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "datasets": datasets,
        "records_across_files": sum(v["records"] for v in datasets.values()),
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    write_json(root / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def cmd_download(args: argparse.Namespace) -> int:
    names = list(SOURCE_SPECS) if args.source == "all" else [args.source]
    for name in names:
        if name == "tokenizers":
            for slug in TOKENIZER_SPECS:
                download_tokenizer(slug, force=args.force)
        else:
            download_source(name, force=args.force)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    manifest = build_workloads()
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    report = audit_workload(Path(args.root))
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_perf_workload",
        description="Build/audit Mac text-model performance workloads")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download pinned public sources/tokenizers")
    d.add_argument("--source", default="all",
                   choices=["all", "dolly", "ultrachat", "longbench", "tokenizers"])
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("build", help="build tokenizer-calibrated JSONL workloads")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("audit", help="audit generated workload")
    a.add_argument("--root", default=str(PERF_ROOT))
    a.set_defaults(func=cmd_audit)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
