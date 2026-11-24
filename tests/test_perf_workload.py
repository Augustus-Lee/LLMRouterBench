# -*- coding: utf-8 -*-
"""Performance workload build/load integration tests (stdlib only)."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.bench_common import load_workload_records, run_benchmark  # noqa: E402
from tools.bench_mlxvlm import build_parser as mlxvlm_parser  # noqa: E402
from tools.bench_splash import build_parser as splash_parser  # noqa: E402
from tools.build_perf_workload import _audit_records, sha256_text  # noqa: E402


def _record(workload_id="controlled-1024-01", target=1024, actual=1024):
    text = "public performance text " * 40
    return {
        "id": workload_id,
        "layer": "controlled",
        "boundary_probe": target >= 262144,
        "target_tokens": target,
        "actual_tokens": actual,
        "tokenizer": "qwen3.6-35b-a3b",
        "tokenizer_model": "Qwen/Qwen3.6-35B-A3B",
        "tokenizer_revision": "pinned-revision",
        "tokenizer_exact": True,
        "text": text,
        "text_sha256": sha256_text(text),
        "source_dataset": "example/public",
        "source_revision": "pinned-data-revision",
        "source_split": "train",
        "source_ids": ["0"],
        "source_license": "Apache-2.0",
        "construction": "prefix_fit_and_repeat_public_text",
        "variant": 0,
        "seed": 42,
        "question_included": False,
        "token_counts": {"qwen3.6-35b-a3b": actual},
        "notes": "performance workload only",
    }


class _FakeClient:
    def chat_nonstream(self, **kwargs):
        return {"success": True}

    def chat_stream(self, **kwargs):
        return {
            "stream": True,
            "prompt_tokens": 1024,
            "completion_tokens": 8,
            "ttft_s": 0.1,
            "prefill_s": None,
            "decode_s": 0.2,
            "prefill_tok_per_s": None,
            "decode_tok_per_s": 40.0,
            "e2e_s": 0.3,
            "timing_source": "client",
            "finish_reason": "stop",
            "error": None,
            "success": True,
        }


class TestWorkloadAudit(unittest.TestCase):
    def test_valid_record(self):
        errors, warnings = _audit_records([_record()], "fixture.jsonl")
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_duplicate_id_rejected(self):
        errors, _ = _audit_records([_record(), _record()], "fixture.jsonl")
        self.assertTrue(any("duplicate id" in e for e in errors))

    def test_empty_text_rejected(self):
        rec = _record()
        rec["text"] = ""
        rec["text_sha256"] = sha256_text("")
        errors, _ = _audit_records([rec], "fixture.jsonl")
        self.assertTrue(any("empty text" in e for e in errors))

    def test_hash_mismatch_rejected(self):
        rec = _record()
        rec["text_sha256"] = "0" * 64
        errors, _ = _audit_records([rec], "fixture.jsonl")
        self.assertTrue(any("text_sha256 mismatch" in e for e in errors))

    def test_token_tolerance_enforced(self):
        rec = _record(actual=900)
        errors, _ = _audit_records([rec], "fixture.jsonl")
        self.assertTrue(any("97%" in e for e in errors))

    def test_256k_must_be_boundary_probe(self):
        rec = _record("controlled-262144-01", 262144, 262144)
        rec["boundary_probe"] = False
        errors, _ = _audit_records([rec], "fixture.jsonl")
        self.assertTrue(any("boundary_probe" in e for e in errors))

    def test_provenance_required(self):
        rec = _record()
        rec["source_license"] = None
        errors, _ = _audit_records([rec], "fixture.jsonl")
        self.assertTrue(any("source_license" in e for e in errors))


class TestWorkloadLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="perf-workload-")
        self.path = os.path.join(self.tmp, "workload.jsonl")
        with open(self.path, "w", encoding="utf-8") as f:
            for rec in (_record(), _record("real-example-8192-01", 8192, 8192)):
                f.write(json.dumps(rec) + "\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def test_filter_by_target_tokens(self):
        records = load_workload_records(self.path, contexts=[8192])
        self.assertEqual([r["id"] for r in records], ["real-example-8192-01"])

    def test_missing_context_is_error(self):
        with self.assertRaises(ValueError):
            load_workload_records(self.path, contexts=[32768])

    def test_adapters_accept_workload_file(self):
        self.assertEqual(
            splash_parser().parse_args(["--model", "m", "--workload-file", self.path]
                                      ).workload_file, self.path)
        self.assertEqual(
            mlxvlm_parser().parse_args(["--model", "m", "--workload-file", self.path]
                                      ).workload_file, self.path)

    def test_run_benchmark_records_provenance_and_unique_filename(self):
        out = os.path.join(self.tmp, "raw")
        args = SimpleNamespace(
            out=out, contexts=[1024], engine="splash", engine_version="test",
            model="m", port=1, max_output_tokens=8, draft_model=None,
            kv_format=None, state_format=None, weight_format="mlx",
            quantization="4bit", temperature=0.0, top_p=1.0, seed=42,
            cold_load_s=None, warmup=0, repeat=1, revision_or_digest=None,
            server_pid=None, monitor_disk=None, workload_file=self.path,
        )
        paths = run_benchmark(args, _FakeClient())
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith("splash_controlled-1024-01.json"))
        with open(paths[0], encoding="utf-8") as f:
            result = json.load(f)
        self.assertEqual(result["workload_id"], "controlled-1024-01")
        self.assertEqual(result["source_license"], "Apache-2.0")
        self.assertEqual(result["workload_target_tokens"], 1024)
        self.assertEqual(result["context_tokens"], 1024)
        with open(self.path, "rb") as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(result["workload_file_sha256"], expected)


if __name__ == "__main__":
    unittest.main()
