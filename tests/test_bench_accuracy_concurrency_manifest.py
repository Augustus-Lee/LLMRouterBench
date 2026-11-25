# -*- coding: utf-8 -*-
"""bench_accuracy_concurrency 的 manifest 数据源逻辑测试。"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.bench_accuracy_concurrency import (  # noqa: E402
    build_parser,
    build_queries_from_manifest,
    load_manifest,
)


class TestLoadManifest(unittest.TestCase):
    def test_load(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write('{"dataset_id": "gpqa", "record_index": 3}\n')
            f.write('{"dataset_id": "mmlu", "record_index": 7}\n')
            path = f.name
        try:
            entries = load_manifest(path)
            self.assertEqual(entries, [
                {"dataset_id": "gpqa", "record_index": 3},
                {"dataset_id": "mmlu", "record_index": 7},
            ])
            self.assertIsInstance(entries[0]["record_index"], int)
        finally:
            os.unlink(path)

    def test_missing_raises(self):
        with self.assertRaises(SystemExit):
            load_manifest("/no/such/manifest.jsonl")


class TestBuildQueriesFromManifest(unittest.TestCase):
    def setUp(self):
        self.items = {
            "gpqa": [{"prompt": "p0", "answer": "A"}, {"prompt": "p1", "answer": "B"}],
            "mmlu": [{"prompt": "m0", "answer": "C"}],
        }

    def test_sorted_and_mapped(self):
        manifest = [
            {"dataset_id": "gpqa", "record_index": 1},
            {"dataset_id": "gpqa", "record_index": 0},
            {"dataset_id": "mmlu", "record_index": 0},
        ]
        qs = build_queries_from_manifest(manifest, self.items)
        self.assertEqual([(q["dataset_id"], q["record_index"]) for q in qs],
                         [("gpqa", 0), ("gpqa", 1), ("mmlu", 0)])
        self.assertEqual([q["prompt"] for q in qs], ["p0", "p1", "m0"])
        self.assertEqual([q["ground_truth"] for q in qs], ["A", "B", "C"])

    def test_invalid_index_skipped(self):
        manifest = [
            {"dataset_id": "gpqa", "record_index": 0},
            {"dataset_id": "gpqa", "record_index": 99},   # 越界跳过
            {"dataset_id": "mmlu", "record_index": 1},    # 越界跳过
        ]
        qs = build_queries_from_manifest(manifest, self.items)
        self.assertEqual(len(qs), 1)
        self.assertEqual(qs[0]["dataset_id"], "gpqa")

    def test_empty_prompt_skipped(self):
        items = {"gpqa": [{"prompt": "", "answer": "A"}]}
        manifest = [{"dataset_id": "gpqa", "record_index": 0}]
        self.assertEqual(build_queries_from_manifest(manifest, items), [])

    def test_unknown_dataset_skipped(self):
        manifest = [{"dataset_id": "nope", "record_index": 0}]
        self.assertEqual(build_queries_from_manifest(manifest, self.items), [])


class TestParserManifest(unittest.TestCase):
    def test_default_query_source_still_data(self):
        args = build_parser().parse_args(["--engine", "splash-qwen38"])
        self.assertEqual(args.query_source, "data")
        self.assertIsNone(args.manifest)

    def test_manifest_mode(self):
        args = build_parser().parse_args(
            ["--engine", "splash-qwen38", "--query-source", "manifest",
             "--manifest", "results/sample_882.jsonl"])
        self.assertEqual(args.query_source, "manifest")
        self.assertEqual(args.manifest, "results/sample_882.jsonl")


if __name__ == "__main__":
    unittest.main()
