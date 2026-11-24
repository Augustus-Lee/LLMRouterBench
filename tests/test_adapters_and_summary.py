# -*- coding: utf-8 -*-
"""适配器 CLI 默认值与 summarize 汇总逻辑测试。"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.bench_splash import build_parser as splash_parser  # noqa: E402
from tools.bench_mlxvlm import build_parser as mlxvlm_parser  # noqa: E402
from tools.summarize import load_records, render_table, engine_table_letter  # noqa: E402


class TestSplashDefaults(unittest.TestCase):
    def test_defaults(self):
        args = splash_parser().parse_args(["--model", "mlx-community/Qwen3.6-35B-A3B-4bit"])
        self.assertEqual(args.engine, "splash")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 18081)
        self.assertEqual(args.contexts, [1024, 8192, 32768, 65536, 131072])
        self.assertEqual(args.warmup, 1)
        self.assertEqual(args.repeat, 3)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.top_p, 1.0)

    def test_custom_contexts(self):
        args = splash_parser().parse_args(
            ["--model", "m", "--contexts", "1024,8192", "--repeat", "5"])
        self.assertEqual(args.contexts, [1024, 8192])
        self.assertEqual(args.repeat, 5)


class TestMlxVlmDefaults(unittest.TestCase):
    def test_defaults(self):
        args = mlxvlm_parser().parse_args(["--model", "/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit"])
        self.assertEqual(args.engine, "mlx-vlm")
        self.assertEqual(args.port, 18091)
        self.assertEqual(args.contexts, [1024, 8192, 32768, 65536, 131072])


def _make_record(engine, model, context, decode_vals, ttft_vals, rss_gib):
    return {
        "engine": engine,
        "engine_version": "1.5.0" if engine == "splash" else "0.8.0",
        "port": 18081 if engine == "splash" else 18091,
        "model": model,
        "weight_format": "mlx",
        "quantization": "4bit-affine",
        "draft_model": "draft" if engine == "splash" else None,
        "kv_format": "int8" if engine == "splash" else None,
        "hardware": {"host": "Jiuwen-MacMini", "chip": "Apple M5 Pro",
                     "model_identifier": "Mac17,16", "memory_gib": 64, "os_version": "27.2"},
        "context_tokens": context,
        "max_output_tokens": 256,
        "temperature": 0.0,
        "seed": 42,
        "cold_load_s": 30.0,
        "repeats": [
            {"repeat": i, "stream": True, "prompt_tokens": context,
             "completion_tokens": 256, "ttft_s": t, "prefill_s": None,
             "decode_s": 10.0, "decode_tok_per_s": 256 / 10.0,
             "e2e_s": 10.0 + t, "timing_source": "client",
             "finish_reason": "stop", "error": None, "success": True}
            for i, t in enumerate(ttft_vals)
        ],
        "memory": {"peak_process_gib": rss_gib, "wired_gib": None,
                   "swap_delta_gib": None, "memory_pressure": None},
        "ssd_read_mib_s": None,
        "error_count": 0,
        "success": True,
        "notes": "",
    }


class TestSummarize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="benchsum-")
        self.raw = os.path.join(self.tmp, "raw")
        os.makedirs(self.raw)
        recs = [
            _make_record("splash", "mlx-community/Qwen3.6-35B-A3B-4bit", 1024, [1.0], [0.1], 20.0),
            _make_record("splash", "mlx-community/Qwen3.6-35B-A3B-4bit", 8192, [1.0], [0.2], 21.0),
            _make_record("mlx-vlm", "gemma-4-26b-a4b-it-4bit", 1024, [1.0], [0.3], 15.0),
        ]
        for i, rec in enumerate(recs):
            name = f"{rec['engine']}_{rec['context_tokens']}.json"
            with open(os.path.join(self.raw, name), "w") as f:
                json.dump(rec, f)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_load_records(self):
        recs = load_records(self.raw)
        self.assertEqual(len(recs), 3)

    def test_table_letter(self):
        self.assertEqual(engine_table_letter("mlx-community/Qwen3.6-35B-A3B-4bit"), "B")
        self.assertEqual(engine_table_letter("gemma-4-26b-a4b-it-4bit"), "C")
        self.assertEqual(engine_table_letter("unknown-model"), "extra")

    def test_render_table_contains_stats(self):
        recs = load_records(self.raw)
        splash = [r for r in recs if r["engine"] == "splash"]
        md = render_table("splash", splash)
        self.assertIn("Splash", md)
        self.assertIn("1024", md)
        self.assertIn("8192", md)
        self.assertIn("Decode tok/s", md)
        # decode tok/s = 256/10 = 25.6
        self.assertIn("25.6", md)
        self.assertIn("峰值 RSS", md)


if __name__ == "__main__":
    unittest.main()
