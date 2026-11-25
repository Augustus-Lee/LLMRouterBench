# -*- coding: utf-8 -*-
"""bench_accuracy_concurrency 纯逻辑测试：query 构造、判分记录、档位汇总、参数解析。"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.bench_accuracy_concurrency import (  # noqa: E402
    build_queries,
    compute_stage_summary,
    make_request_record,
    make_workload_event,
    parse_concurrency,
    per_dataset_accuracy,
    build_parser,
)


def _rec(ds, idx, ans):
    return SimpleNamespace(dataset_id=ds, record_index=idx, prompt=f"q-{ds}-{idx}",
                           ground_truth=ans, origin_query=f"oq-{ds}-{idx}")


class TestBuildQueries(unittest.TestCase):
    def test_sorted_by_dataset_order_then_index(self):
        recs = [_rec("mmlu", 3, "B"), _rec("gpqa", 1, "A"), _rec("gpqa", 0, "C")]
        qs = build_queries(recs)
        self.assertEqual([(q["dataset_id"], q["record_index"]) for q in qs],
                         [("gpqa", 0), ("gpqa", 1), ("mmlu", 3)])

    def test_only_filter(self):
        recs = [_rec("gpqa", 0, "A"), _rec("mmlu", 0, "B")]
        qs = build_queries(recs, only={"gpqa"})
        self.assertEqual([q["dataset_id"] for q in qs], ["gpqa"])

    def test_limit_per_dataset(self):
        recs = [_rec("gpqa", i, "A") for i in range(5)]
        qs = build_queries(recs, limit_per_dataset=3)
        self.assertEqual(len(qs), 3)
        self.assertEqual([q["record_index"] for q in qs], [0, 1, 2])

    def test_prompt_and_ground_truth_preserved(self):
        q = build_queries([_rec("gpqa", 7, "D")])[0]
        self.assertEqual(q["prompt"], "q-gpqa-7")
        self.assertEqual(q["ground_truth"], "D")
        self.assertEqual(q["record_index"], 7)


class TestMakeRequestRecord(unittest.TestCase):
    def test_success_scored(self):
        raw = {"success": True, "content": "A", "prompt_tokens": 10,
               "completion_tokens": 5, "decode_tok_per_s": 8.0,
               "prefill_tok_per_s": 20.0, "e2e_s": 1.2, "finish_reason": "stop"}
        eval_result = {"is_correct": True, "prediction": "A"}
        r = make_request_record({"dataset_id": "gpqa", "record_index": 0,
                                 "prompt": "q", "ground_truth": "A"},
                                raw, eval_result, "nonstream", 0, 0)
        self.assertTrue(r["response"] == "A")
        self.assertTrue(r["is_correct"])
        self.assertEqual(r["prediction"], "A")
        self.assertIsNone(r["fail_reason"])
        self.assertTrue(r["success"])

    def test_request_failure(self):
        raw = {"success": False, "content": None, "error": "connection"}
        r = make_request_record({"dataset_id": "gpqa", "record_index": 0,
                                 "prompt": "q", "ground_truth": "A"},
                                raw, None, "nonstream", 0, 1)
        self.assertFalse(r["success"])
        self.assertIsNone(r["is_correct"])
        self.assertEqual(r["fail_reason"], "connection")
        self.assertEqual(r["retry_count"], 1)

    def test_scoring_error_marks_unjudgeable(self):
        raw = {"success": True, "content": "strange"}  # 请求成功但判分失败
        r = make_request_record({"dataset_id": "math500", "record_index": 0,
                                 "prompt": "q", "ground_truth": "42"},
                                raw, None, "nonstream", 0, 0)
        self.assertTrue(r["success"])
        self.assertIsNone(r["is_correct"])
        self.assertEqual(r["fail_reason"], "scoring_error")

    def test_nonbool_is_correct_kept_none(self):
        raw = {"success": True, "content": "x"}
        r = make_request_record({"dataset_id": "gpqa", "record_index": 0,
                                 "prompt": "q", "ground_truth": "A"},
                                raw, {"is_correct": None, "prediction": None},
                                "nonstream", 0, 0)
        self.assertIsNone(r["is_correct"])


class TestPerDatasetAccuracy(unittest.TestCase):
    def test_grouping(self):
        reqs = [
            {"dataset_id": "gpqa", "success": True, "is_correct": True},
            {"dataset_id": "gpqa", "success": True, "is_correct": False},
            {"dataset_id": "mmlu", "success": True, "is_correct": True},
            {"dataset_id": "mmlu", "success": False, "is_correct": None},
        ]
        out = per_dataset_accuracy(reqs)
        self.assertEqual(out["gpqa"]["n_scored"], 2)
        self.assertAlmostEqual(out["gpqa"]["accuracy"], 0.5)
        self.assertEqual(out["mmlu"]["n_scored"], 1)
        self.assertAlmostEqual(out["mmlu"]["accuracy"], 1.0)


class TestComputeStageSummary(unittest.TestCase):
    def test_accuracy_and_timing(self):
        reqs = [
            {"success": True, "is_correct": True, "decode_tok_per_s": 10.0,
             "ttft_s": None, "finish_reason": "stop", "dataset_id": "gpqa"},
            {"success": True, "is_correct": True, "decode_tok_per_s": 20.0,
             "ttft_s": None, "finish_reason": "stop", "dataset_id": "gpqa"},
            {"success": True, "is_correct": False, "decode_tok_per_s": 30.0,
             "ttft_s": None, "finish_reason": "stop", "dataset_id": "mmlu"},
            {"success": False, "is_correct": None, "decode_tok_per_s": None,
             "ttft_s": None, "finish_reason": None, "dataset_id": "mmlu"},
        ]
        s = compute_stage_summary(reqs, wall_s=10.0, concurrency=2)
        self.assertEqual(s["concurrency"], 2)
        self.assertAlmostEqual(s["accuracy"], 2 / 3)   # 2/3 判分正确
        self.assertEqual(s["n_scored"], 3)
        self.assertEqual(s["success_requests"], 3)
        self.assertEqual(s["error_requests"], 1)
        self.assertEqual(s["success_rate"], 0.75)
        self.assertAlmostEqual(s["decode_tok_per_s_mean"], 20.0)
        self.assertIsNone(s["ttft_s_mean"])  # 非流式无 TTFT
        # 每数据集正确率
        self.assertAlmostEqual(per_dataset_accuracy(reqs)["gpqa"]["accuracy"], 1.0)
        self.assertAlmostEqual(per_dataset_accuracy(reqs)["mmlu"]["accuracy"], 0.0)

    def test_phase_effective_compute_and_bandwidth(self):
        reqs = [
            {"success": True, "is_correct": True, "dataset_id": "gpqa",
             "prompt_tokens": 1000, "completion_tokens": 100,
             "ttft_s": 0.2, "decode_tok_per_s": 20.0},
        ]
        s = compute_stage_summary(
            reqs, wall_s=10.0, concurrency=1,
            theoretical_tflops=100.0,
            official_bandwidth_gb_s=200.0,
            prefill_flops_per_token=2e9,
            decode_flops_per_token=4e9,
            prefill_bytes_per_token=1e9,
            decode_bytes_per_token=2e9,
        )
        self.assertAlmostEqual(s["input_tokens_per_s"], 100.0)
        self.assertAlmostEqual(s["output_tokens_per_s"], 10.0)
        self.assertAlmostEqual(s["prefill_effective_tflops"], 0.2)
        self.assertAlmostEqual(s["decode_effective_tflops"], 0.04)
        self.assertAlmostEqual(s["effective_tflops"], 0.24)
        self.assertAlmostEqual(s["prefill_effective_bandwidth_gb_s"], 100.0)
        self.assertAlmostEqual(s["decode_effective_bandwidth_gb_s"], 20.0)
        self.assertAlmostEqual(s["effective_bandwidth_gb_s"], 120.0)
        self.assertAlmostEqual(s["tflops_utilization"], 0.0024)
        self.assertAlmostEqual(s["bandwidth_utilization"], 0.6)

    def test_missing_compute_inputs_stay_null(self):
        reqs = [{"success": True, "is_correct": True, "dataset_id": "gpqa",
                 "prompt_tokens": 1000, "completion_tokens": 100}]
        s = compute_stage_summary(reqs, wall_s=10.0, concurrency=1)
        self.assertIsNone(s["prefill_effective_tflops"])
        self.assertIsNone(s["decode_effective_tflops"])
        self.assertIsNone(s["effective_tflops"])
        self.assertIsNone(s["prefill_effective_bandwidth_gb_s"])
        self.assertIsNone(s["effective_bandwidth_gb_s"])


class TestWorkloadEvent(unittest.TestCase):
    def test_successful_request_has_token_and_phase_timing_fields(self):
        record = {
            "success": True, "mode": "stream", "prompt_tokens": 100,
            "completion_tokens": 20, "ttft_s": 0.25, "e2e_s": 1.25,
        }
        event = make_workload_event(record, concurrency=4, wall_time=123.5)
        self.assertEqual(event["wall_time"], 123.5)
        self.assertEqual(event["concurrency"], 4)
        self.assertEqual(event["prompt_tokens"], 100)
        self.assertEqual(event["completion_tokens"], 20)
        self.assertEqual(event["ttft_s"], 0.25)

    def test_failed_request_is_not_counted_as_workload(self):
        record = {"success": False, "prompt_tokens": None,
                  "completion_tokens": None}
        self.assertIsNone(make_workload_event(record, concurrency=1, wall_time=1.0))


class TestParseConcurrency(unittest.TestCase):
    def test_csv(self):
        self.assertEqual(parse_concurrency("1,2,4"), [1, 2, 4])
        self.assertEqual(parse_concurrency(" 8 ,16 "), [8, 16])


class TestParserDefaults(unittest.TestCase):
    def test_defaults(self):
        p = build_parser()
        args = p.parse_args(["--engine", "splash-qwen36"])
        self.assertEqual(args.engine, "splash-qwen36")
        self.assertEqual(args.concurrency, [1, 2, 4])
        self.assertEqual(args.mode, "nonstream")
        self.assertEqual(args.repeat, 1)
        self.assertEqual(args.limit_per_dataset, 0)
        self.assertEqual(args.max_attempts, 3)
        self.assertIsNone(args.live_counters_file)
        self.assertEqual(args.workload_window_s, 15.0)
        self.assertIsNone(args.theoretical_tflops)
        self.assertIsNone(args.prefill_flops_per_token)
        self.assertIsNone(args.decode_bytes_per_token)


if __name__ == "__main__":
    unittest.main()
