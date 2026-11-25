# -*- coding: utf-8 -*-
"""并发熔断阈值逻辑测试：RSS / swap / 错误率触发与豁免。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.bench_accuracy_concurrency import (  # noqa: E402
    build_parser,
    compute_stage_summary,
    should_abort,
)


def _summary(rss=None, swap=None, error_rate=None):
    return {"memory": {"peak_process_gib": rss, "swap_delta_gib": swap},
            "error_rate": error_rate}


class TestShouldAbort(unittest.TestCase):
    def test_under_thresholds_no_abort(self):
        self.assertEqual(should_abort(_summary(50.0, 1.0, 0.1), 55.0, 4.0, 0.3), [])

    def test_rss_over(self):
        reasons = should_abort(_summary(60.0, 1.0, 0.1), 55.0, 4.0, 0.3)
        self.assertTrue(any("rss" in r for r in reasons))

    def test_swap_over(self):
        reasons = should_abort(_summary(50.0, 5.0, 0.1), 55.0, 4.0, 0.3)
        self.assertTrue(any("swap" in r for r in reasons))

    def test_error_rate_over(self):
        reasons = should_abort(_summary(50.0, 1.0, 0.5), 55.0, 4.0, 0.3)
        self.assertTrue(any("error_rate" in r for r in reasons))

    def test_multiple_reasons(self):
        reasons = should_abort(_summary(60.0, 5.0, 0.5), 55.0, 4.0, 0.3)
        self.assertEqual(len(reasons), 3)

    def test_none_metric_no_trigger(self):
        # memory 字段缺失 / 指标 None 不应误触发
        self.assertEqual(should_abort(_summary(None, None, None), 55.0, 4.0, 0.3), [])
        self.assertEqual(should_abort({"memory": None, "error_rate": None}, 55.0, 4.0, 0.3), [])

    def test_none_threshold_disabled(self):
        # 阈值显式关闭（None）时不触发
        self.assertEqual(should_abort(_summary(99.0, 99.0, 0.99), None, None, None), [])


class TestComputeSummaryErrorRate(unittest.TestCase):
    def test_error_rate_computed(self):
        reqs = [
            {"success": True, "is_correct": True, "dataset_id": "gpqa",
             "ttft_s": None, "decode_tok_per_s": 10.0},
            {"success": False, "is_correct": None, "dataset_id": "gpqa",
             "ttft_s": None, "decode_tok_per_s": None},
        ]
        s = compute_stage_summary(reqs, wall_s=10.0, concurrency=1)
        self.assertEqual(s["error_rate"], 0.5)
        self.assertEqual(s["success_rate"], 0.5)


class TestParserDefaults(unittest.TestCase):
    def test_breaker_defaults(self):
        args = build_parser().parse_args(["--engine", "splash-qwen38"])
        self.assertEqual(args.max_rss_gib, 55.0)
        self.assertEqual(args.max_swap_gib, 4.0)
        self.assertEqual(args.max_error_rate, 0.3)

    def test_breaker_override(self):
        args = build_parser().parse_args(
            ["--engine", "splash-qwen38", "--max-rss-gib", "50", "--max-error-rate", "0.2"])
        self.assertEqual(args.max_rss_gib, 50.0)
        self.assertEqual(args.max_error_rate, 0.2)


if __name__ == "__main__":
    unittest.main()


class TestShardQueries(unittest.TestCase):
    def test_round_robin(self):
        from tools.bench_accuracy_concurrency import shard_queries
        qs = list(range(8))
        self.assertEqual(shard_queries(qs, 4, 0), [0, 4])
        self.assertEqual(shard_queries(qs, 4, 1), [1, 5])
        self.assertEqual(shard_queries(qs, 4, 2), [2, 6])
        self.assertEqual(shard_queries(qs, 4, 3), [3, 7])

    def test_single_shard(self):
        from tools.bench_accuracy_concurrency import shard_queries
        self.assertEqual(shard_queries([10, 20, 30], 1, 0), [10, 20, 30])

    def test_index_out_of_range_empty(self):
        from tools.bench_accuracy_concurrency import shard_queries
        self.assertEqual(shard_queries([1, 2, 3], 2, 99), [])


class TestShardParserFlag(unittest.TestCase):
    def test_flag_off_by_default(self):
        args = build_parser().parse_args(["--engine", "splash-qwen38"])
        self.assertFalse(args.shard_across_levels)

    def test_flag_on(self):
        args = build_parser().parse_args(
            ["--engine", "splash-qwen38", "--shard-across-levels"])
        self.assertTrue(args.shard_across_levels)


if __name__ == "__main__":
    unittest.main()
