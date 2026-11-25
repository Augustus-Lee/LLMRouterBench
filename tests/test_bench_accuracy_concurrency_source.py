# -*- coding: utf-8 -*-
"""bench_accuracy_concurrency 数据源与并发容量钳制逻辑测试。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.bench_accuracy_concurrency import (  # noqa: E402
    build_queries_from_items,
    build_parser,
    clamp_concurrency,
)


class TestBuildQueriesFromItems(unittest.TestCase):
    def test_basic(self):
        items = [{"prompt": "p0", "answer": "A"}, {"prompt": "p1", "answer": "B"}]
        qs = build_queries_from_items("gpqa", items)
        self.assertEqual([q["record_index"] for q in qs], [0, 1])
        self.assertEqual([q["ground_truth"] for q in qs], ["A", "B"])
        self.assertEqual([q["dataset_id"] for q in qs], ["gpqa", "gpqa"])

    def test_skip_empty_prompt(self):
        items = [{"prompt": "", "answer": "A"}, {"prompt": "p1", "answer": "B"}]
        qs = build_queries_from_items("gpqa", items)
        self.assertEqual(len(qs), 1)
        self.assertEqual(qs[0]["record_index"], 0)  # 空 prompt 不计入索引

    def test_limit(self):
        items = [{"prompt": f"p{i}", "answer": "A"} for i in range(5)]
        qs = build_queries_from_items("gpqa", items, limit_per_dataset=3)
        self.assertEqual(len(qs), 3)


class TestClampConcurrency(unittest.TestCase):
    def test_clamp(self):
        kept, skipped = clamp_concurrency([1, 2, 4, 8, 16], 4)
        self.assertEqual(kept, [1, 2, 4])
        self.assertEqual(skipped, [8, 16])

    def test_no_service_limit(self):
        kept, skipped = clamp_concurrency([1, 2, 4], None)
        self.assertEqual(kept, [1, 2, 4])
        self.assertEqual(skipped, [])

    def test_service_above_ladder(self):
        kept, skipped = clamp_concurrency([1, 2], 99)
        self.assertEqual(kept, [1, 2])
        self.assertEqual(skipped, [])


class TestParserDefaults(unittest.TestCase):
    def test_query_source_defaults_to_data(self):
        args = build_parser().parse_args(["--engine", "splash-qwen38"])
        self.assertEqual(args.query_source, "data")  # results/bench 缺失时可先跑
        self.assertIsNone(args.service_max_concurrency)
        self.assertEqual(args.limit_per_dataset, 0)

    def test_query_source_882(self):
        args = build_parser().parse_args(["--engine", "splash-qwen38", "--query-source", "882"])
        self.assertEqual(args.query_source, "882")

    def test_service_max_concurrency(self):
        args = build_parser().parse_args(
            ["--engine", "splash-qwen38", "--service-max-concurrency", "4"])
        self.assertEqual(args.service_max_concurrency, 4)


if __name__ == "__main__":
    unittest.main()
