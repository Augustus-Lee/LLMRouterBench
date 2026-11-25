# -*- coding: utf-8 -*-
"""bench_common 纯逻辑单元测试：prompt 合成、统计、macOS 解析器、JSON schema。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.bench_common import (  # noqa: E402
    estimate_tokens,
    extract_server_timings,
    make_run_record,
    parse_swapusage,
    parse_vm_stat_wired_pages,
    swap_delta_mib,
    summarize_repeats,
    synthesize_prompt,
)


class TestEstimateTokens(unittest.TestCase):
    def test_english_words_count_as_tokens(self):
        text = "the quick brown fox jumps over the lazy dog"  # 9 words
        self.assertEqual(estimate_tokens(text), 9)

    def test_cjk_chars_count_individually(self):
        text = "你好世界"  # 4 CJK chars
        self.assertEqual(estimate_tokens(text), 4)

    def test_mixed(self):
        text = "hello 世界"  # 1 word + 2 cjk
        self.assertEqual(estimate_tokens(text), 3)


class TestSynthesizePrompt(unittest.TestCase):
    def test_deterministic_with_same_seed(self):
        a = synthesize_prompt(1000, seed=42)
        b = synthesize_prompt(1000, seed=42)
        self.assertEqual(a, b)

    def test_different_seed_different_text(self):
        a = synthesize_prompt(1000, seed=1)
        b = synthesize_prompt(1000, seed=2)
        self.assertNotEqual(a, b)

    def test_target_length_within_tolerance(self):
        for target in (1024, 8192):
            text = synthesize_prompt(target, seed=42)
            est = estimate_tokens(text)
            self.assertGreater(est, target * 0.90, f"target={target} est={est}")
            self.assertLess(est, target * 1.10, f"target={target} est={est}")

    def test_monotonic_across_ladder(self):
        e1 = estimate_tokens(synthesize_prompt(1024, seed=42))
        e2 = estimate_tokens(synthesize_prompt(8192, seed=42))
        e3 = estimate_tokens(synthesize_prompt(65536, seed=42))
        self.assertLess(e1, e2)
        self.assertLess(e2, e3)


class TestSummarizeRepeats(unittest.TestCase):
    def test_known_values(self):
        stats = summarize_repeats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        self.assertEqual(stats["n"], 10)
        self.assertAlmostEqual(stats["mean"], 5.5)
        self.assertAlmostEqual(stats["median"], 5.5)
        # nearest-rank p95 of 10 samples -> ceil(0.95*10)=10 -> 10.0
        self.assertAlmostEqual(stats["p95"], 10.0)
        self.assertAlmostEqual(stats["min"], 1.0)
        self.assertAlmostEqual(stats["max"], 10.0)

    def test_p95_nearest_rank_small_list(self):
        stats = summarize_repeats([1.0, 2.0, 3.0])
        # ceil(0.95*3)=3 -> 3.0
        self.assertAlmostEqual(stats["p95"], 3.0)

    def test_empty_returns_nones(self):
        stats = summarize_repeats([])
        self.assertIsNone(stats["mean"])
        self.assertIsNone(stats["p95"])
        self.assertEqual(stats["n"], 0)

    def test_none_values_filtered(self):
        stats = summarize_repeats([1.0, None, 3.0])
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["mean"], 2.0)


class TestMakeRunRecord(unittest.TestCase):
    def test_schema_has_null_not_zero(self):
        rec = make_run_record(
            engine="splash",
            engine_version="1.5.0",
            model="mlx-community/Qwen3.6-35B-A3B-4bit",
            port=18081,
            context_tokens=8192,
            max_output_tokens=256,
            draft_model="Qwen3.6-35B-A3B-DFlash2",
            kv_format="int8",
        )
        self.assertEqual(rec["engine"], "splash")
        self.assertIsNone(rec["cold_load_s"])
        self.assertIsNone(rec["ssd_read_mib_s"])
        self.assertEqual(rec["memory"]["peak_process_gib"], None)
        self.assertEqual(rec["memory"]["swap_delta_gib"], None)
        self.assertEqual(rec["repeats"], [])
        self.assertIn("hardware", rec)

    def test_mlxvlm_record_has_no_draft(self):
        rec = make_run_record(
            engine="mlx-vlm",
            engine_version="0.8.0",
            model="/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit",
            port=18091,
            context_tokens=8192,
            max_output_tokens=256,
            draft_model=None,
            kv_format=None,
        )
        self.assertIsNone(rec["draft_model"])
        self.assertIsNone(rec["kv_format"])


class TestMacParsers(unittest.TestCase):
    def test_parse_swapusage_megabytes(self):
        text = "total: 1024.00M used: 12.50M free: 1011.50M (encrypted)"
        out = parse_swapusage(text)
        self.assertAlmostEqual(out["total_mib"], 1024.0)
        self.assertAlmostEqual(out["used_mib"], 12.5)
        self.assertAlmostEqual(out["free_mib"], 1011.5)

    def test_parse_swapusage_gigabytes(self):
        text = "total: 2.00Gi used: 0.50Gi free: 1.50Gi"
        out = parse_swapusage(text)
        self.assertAlmostEqual(out["used_mib"], 512.0)

    def test_parse_swapusage_garbage(self):
        self.assertIsNone(parse_swapusage("not a swap line"))

    def test_parse_vm_stat_wired(self):
        text = "Pages wired down: 123456.\nPages active: 999."
        self.assertEqual(parse_vm_stat_wired_pages(text), 123456)

    def test_parse_vm_stat_missing(self):
        self.assertIsNone(parse_vm_stat_wired_pages("nothing here"))

    def test_swap_delta(self):
        before = {"used_mib": 100.0}
        after = {"used_mib": 250.0}
        self.assertAlmostEqual(swap_delta_mib(before, after), 150.0)

    def test_swap_delta_none_inputs(self):
        self.assertIsNone(swap_delta_mib(None, {"used_mib": 1.0}))
        self.assertIsNone(swap_delta_mib({"used_mib": 1.0}, None))


class TestExtractServerTimings(unittest.TestCase):
    def test_llamacpp_style_ms(self):
        data = {"timings": {"prompt_ms": 1000.0, "generation_ms": 4000.0}}
        prefill, decode, source = extract_server_timings(data)
        self.assertAlmostEqual(prefill, 1.0)
        self.assertAlmostEqual(decode, 4.0)
        self.assertEqual(source, "server")

    def test_nanosecond_style(self):
        data = {"prompt_eval_duration": 2_000_000_000, "eval_duration": 5_000_000_000,
                "prompt_eval_count": 100, "eval_count": 50}
        prefill, decode, source = extract_server_timings(data)
        self.assertAlmostEqual(prefill, 2.0)
        self.assertAlmostEqual(decode, 5.0)
        self.assertEqual(source, "server")

    def test_no_timings_returns_none(self):
        prefill, decode, source = extract_server_timings({"usage": {"prompt_tokens": 10}})
        self.assertIsNone(prefill)
        self.assertIsNone(decode)
        self.assertIsNone(source)


if __name__ == "__main__":
    unittest.main()


class TestExtractServerTtft(unittest.TestCase):
    def test_splash_metrics_nested(self):
        from tools.bench_common import _extract_server_ttft
        data = {"metrics": {"request_latency": {"ttft_ms": 129.56}}}
        self.assertAlmostEqual(_extract_server_ttft(data), 0.12956)

    def test_fallback_start_to_first_token(self):
        from tools.bench_common import _extract_server_ttft
        data = {"metrics": {"request_latency": {"start_to_first_token_ms": 114.9}}}
        self.assertAlmostEqual(_extract_server_ttft(data), 0.1149)

    def test_top_level_request_latency(self):
        from tools.bench_common import _extract_server_ttft
        data = {"request_latency": {"ttft_ms": 200.0}}
        self.assertAlmostEqual(_extract_server_ttft(data), 0.2)

    def test_timings_ttft(self):
        from tools.bench_common import _extract_server_ttft
        data = {"timings": {"ttft_ms": 50.0}}
        self.assertAlmostEqual(_extract_server_ttft(data), 0.05)

    def test_missing_returns_none(self):
        from tools.bench_common import _extract_server_ttft
        self.assertIsNone(_extract_server_ttft({}))
