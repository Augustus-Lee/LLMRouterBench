# -*- coding: utf-8 -*-
"""monitor_machine：powermetrics 分块解析、实时跟随、有效算力/带宽推导。"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.monitor_machine import (  # noqa: E402
    PowermetricsTail,
    WorkloadCounterRing,
    WorkloadCounterTail,
    derive_effective_rates,
    build_parser,
    build_final_metrics_row,
    derive_phase_effective_rates,
    parse_powermetrics_text,
    select_compute_snapshot,
    summarize_workload_window,
)

SAMPLE_1 = """*** Sampled system activity (Wed Nov 25 16:00:00 2026 +0800) (2000.00ms elapsed) ***
GPU active residency:  37.50%
GPU HW active frequency: 1200 MHz
GPU Power: 1530 mW
ANE active residency:  2.25%
ANE HW active frequency: 600 MHz
ANE Power: 18 mW
Thermal pressure: Nominal
"""

SAMPLE_2 = """*** Sampled system activity (Wed Nov 25 16:00:02 2026 +0800) (2000.00ms elapsed) ***
GPU active residency:  82.00%
GPU HW active frequency: 1590 MHz
GPU Power: 6200 mW
ANE active residency:  0.00%
ANE HW active frequency: 0 MHz
ANE Power: 0 mW
Thermal pressure: Fair
"""


class TestParsePowermetrics(unittest.TestCase):
    def test_multiple_samples_are_aligned_per_block(self):
        rows = parse_powermetrics_text(SAMPLE_1 + SAMPLE_2)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["gpu_usage_percent"], 37.5)
        self.assertAlmostEqual(rows[0]["gpu_power_mw"], 1530.0)
        self.assertEqual(rows[0]["thermal_pressure"], "Nominal")
        self.assertAlmostEqual(rows[1]["gpu_usage_percent"], 82.0)
        self.assertAlmostEqual(rows[1]["npu_usage_percent"], 0.0)
        self.assertEqual(rows[1]["thermal_pressure"], "Fair")

    def test_missing_fields_stay_none(self):
        rows = parse_powermetrics_text(
            "*** Sampled system activity (x) (1000ms elapsed) ***\n"
            "GPU active residency: 10%\n")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["gpu_usage_percent"], 10.0)
        self.assertIsNone(rows[0]["gpu_power_mw"])
        self.assertIsNone(rows[0]["npu_usage_percent"])
        self.assertIsNone(rows[0]["thermal_pressure"])


class TestPowermetricsTail(unittest.TestCase):
    def test_waits_for_complete_next_block(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "powermetrics.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(SAMPLE_1)
            tail = PowermetricsTail(path)
            # 第一块尚未确定结束，不能提前解析
            self.assertEqual(tail.read(), [])
            with open(path, "a", encoding="utf-8") as f:
                f.write(SAMPLE_2)
            rows = tail.read()
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["gpu_usage_percent"], 37.5)
            rows = tail.read(flush=True)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["gpu_usage_percent"], 82.0)

    def test_start_at_end_skips_preexisting_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "powermetrics.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(SAMPLE_1)
            tail = PowermetricsTail(path, start_at_end=True)
            self.assertEqual(tail.read(), [])
            with open(path, "a", encoding="utf-8") as f:
                f.write(SAMPLE_2)
            rows = tail.read(flush=True)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["gpu_usage_percent"], 82.0)


class TestParserDefaults(unittest.TestCase):
    def test_live_workload_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.workload_window_s, 15.0)
        self.assertIsNone(args.workload_counters_file)
        self.assertFalse(args.start_workload_at_end)
        self.assertIsNone(args.prefill_flops_per_token)
        self.assertIsNone(args.decode_bytes_per_token)


class TestFinalMetricsRow(unittest.TestCase):
    def test_final_drain_preserves_full_system_sample(self):
        last = {
            "timestamp": 10.0,
            "wall_time": 100.0,
            "cpu": {"used_percent": 42.0},
            "memory": {"used_gib_approx": 38.5},
            "swap": {"used_mib": 12.0},
            "disk_read_mib_s": 3.5,
            "gpu": {"gpu_usage_percent": 20.0},
        }
        row = build_final_metrics_row(
            last_sample=last,
            latest_gpu={"gpu_usage_percent": 75.0, "gpu_power_mw": 5000.0},
            compute={"events_in_window": 2, "effective_tflops": 0.5},
            compute_static={"measurement": "static"},
            timestamp=12.5,
            wall_time=102.5,
        )
        self.assertTrue(row["final_drain"])
        self.assertEqual(row["cpu"], last["cpu"])
        self.assertEqual(row["memory"], last["memory"])
        self.assertEqual(row["swap"], last["swap"])
        self.assertEqual(row["disk_read_mib_s"], 3.5)
        self.assertEqual(row["gpu"]["gpu_usage_percent"], 75.0)
        self.assertEqual(row["compute"]["effective_tflops"], 0.5)
        self.assertEqual(row["compute_static"]["measurement"], "static")
        self.assertNotEqual(row["wall_time"], last["wall_time"])


class TestWorkloadCounterTail(unittest.TestCase):
    def test_follows_complete_jsonl_and_keeps_partial_line(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "live_counters.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"wall_time": 100.0, "prompt_tokens": 10}\n'
                        '{"wall_time": 101.0')
            tail = WorkloadCounterTail(path)
            rows = tail.read()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt_tokens"], 10)
            with open(path, "a", encoding="utf-8") as f:
                f.write(', "completion_tokens": 3}\n')
            rows = tail.read()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["completion_tokens"], 3)

    def test_start_at_end_skips_existing_events(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "live_counters.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"wall_time": 100.0, "prompt_tokens": 10}\n')
            tail = WorkloadCounterTail(path, start_at_end=True)
            self.assertEqual(tail.read(), [])
            with open(path, "a", encoding="utf-8") as f:
                f.write('{"wall_time": 101.0, "prompt_tokens": 20}\n')
            self.assertEqual(tail.read()[0]["prompt_tokens"], 20)


class TestWorkloadCounterRing(unittest.TestCase):
    def test_retains_window_plus_slack_and_drops_stale(self):
        ring = WorkloadCounterRing(window_s=10.0, max_events=100)
        for t in (70.0, 91.0, 95.0):
            ring.add({"wall_time": t, "prompt_tokens": 10, "completion_tokens": 1})
        ring.prune(now=100.0)
        self.assertEqual([e["wall_time"] for e in ring.events], [91.0, 95.0])


class TestComputeSnapshot(unittest.TestCase):
    def test_uses_live_values_only_when_window_has_events(self):
        live = {"events_in_window": 2, "effective_tflops": 0.5}
        static = {"events_in_window": 0, "effective_tflops": None}
        self.assertEqual(
            select_compute_snapshot(2, live, static), live)

    def test_empty_window_keeps_static_baseline(self):
        live = {"events_in_window": 0, "effective_tflops": None}
        static = {"events_in_window": 0, "effective_tflops": None}
        self.assertEqual(
            select_compute_snapshot(0, live, static), static)


class TestSummarizeWorkloadWindow(unittest.TestCase):
    def test_sliding_window_rates_and_phase_metrics(self):
        events = [
            {"wall_time": 91.0, "prompt_tokens": 1000, "completion_tokens": 100},
            {"wall_time": 95.0, "prompt_tokens": 100, "completion_tokens": 20},
            {"wall_time": 99.0, "prompt_tokens": 100, "completion_tokens": 20},
        ]
        out = summarize_workload_window(
            events, now=100.0, window_s=10.0,
            theoretical_tflops=100.0, theoretical_bandwidth_gb_s=200.0,
            prefill_flops_per_token=2e9, decode_flops_per_token=4e9,
            prefill_bytes_per_token=1e9, decode_bytes_per_token=2e9)
        self.assertEqual(out["events_in_window"], 3)
        self.assertAlmostEqual(out["input_tokens_per_s"], 120.0)
        self.assertAlmostEqual(out["output_tokens_per_s"], 14.0)
        self.assertAlmostEqual(out["prefill_effective_tflops"], 0.24)
        self.assertAlmostEqual(out["decode_effective_tflops"], 0.056)
        self.assertAlmostEqual(out["prefill_effective_bandwidth_gb_s"], 120.0)
        self.assertAlmostEqual(out["decode_effective_bandwidth_gb_s"], 28.0)
        self.assertAlmostEqual(out["prefill_tflops_utilization"], 0.0024)
        self.assertAlmostEqual(out["decode_bandwidth_utilization"], 0.14)

    def test_stale_or_missing_data_stays_null(self):
        out = summarize_workload_window(
            [{"wall_time": 50.0, "prompt_tokens": 100, "completion_tokens": 20}],
            now=100.0, window_s=10.0)
        self.assertEqual(out["events_in_window"], 0)
        self.assertIsNone(out["input_tokens_per_s"])
        self.assertIsNone(out["output_tokens_per_s"])
        self.assertIsNone(out["decode_effective_tflops"])
        self.assertIsNone(out["decode_effective_bandwidth_gb_s"])

        out = summarize_workload_window(
            [{"wall_time": 99.0}], now=100.0, window_s=10.0)
        self.assertEqual(out["events_in_window"], 1)
        self.assertIsNone(out["input_tokens_per_s"])
        self.assertIsNone(out["output_tokens_per_s"])

        out = summarize_workload_window(
            [{"wall_time": 98.0, "prompt_tokens": 100, "completion_tokens": None},
             {"wall_time": 99.0, "prompt_tokens": 50, "completion_tokens": 10}],
            now=100.0, window_s=10.0)
        self.assertAlmostEqual(out["input_tokens_per_s"], 15.0)
        self.assertIsNone(out["output_tokens_per_s"])


class TestDeriveEffectiveRates(unittest.TestCase):
    def test_formula_and_utilization(self):
        out = derive_effective_rates(
            theoretical_tflops=100.0,
            theoretical_bandwidth_gb_s=200.0,
            tokens_per_s=50.0,
            flops_per_token=2e9,
            bytes_per_token=4e9,
        )
        self.assertAlmostEqual(out["effective_tflops"], 0.1)
        self.assertAlmostEqual(out["tflops_utilization"], 0.001)
        self.assertAlmostEqual(out["effective_bandwidth_gb_s"], 200.0)
        self.assertAlmostEqual(out["bandwidth_utilization"], 1.0)

    def test_missing_inputs_do_not_fabricate(self):
        out = derive_effective_rates(None, None, 20.0, None, None)
        self.assertIsNone(out["effective_tflops"])
        self.assertIsNone(out["effective_bandwidth_gb_s"])
        self.assertIsNone(out["tflops_utilization"])
        self.assertIsNone(out["bandwidth_utilization"])

    def test_phase_formula_does_not_mix_prefill_and_decode(self):
        out = derive_phase_effective_rates(
            theoretical_tflops=100.0,
            theoretical_bandwidth_gb_s=200.0,
            prefill_tokens_per_s=10.0,
            decode_tokens_per_s=5.0,
            prefill_flops_per_token=2e9,
            decode_flops_per_token=4e9,
            prefill_bytes_per_token=1e9,
            decode_bytes_per_token=3e9,
        )
        self.assertAlmostEqual(out["prefill_effective_tflops"], 0.02)
        self.assertAlmostEqual(out["decode_effective_tflops"], 0.02)
        self.assertAlmostEqual(out["prefill_effective_bandwidth_gb_s"], 10.0)
        self.assertAlmostEqual(out["decode_effective_bandwidth_gb_s"], 15.0)
        self.assertAlmostEqual(out["effective_tflops"], 0.04)
        self.assertAlmostEqual(out["effective_bandwidth_gb_s"], 25.0)
        self.assertAlmostEqual(out["tflops_utilization"], 0.0004)
        self.assertAlmostEqual(out["bandwidth_utilization"], 0.125)


if __name__ == "__main__":
    unittest.main()
