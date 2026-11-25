# -*- coding: utf-8 -*-
"""collect_local_882 纯逻辑测试：引擎预设、配置构造、参数解析。

重依赖（openai/httpx/loguru/datasets）在本机缺失，故被测模块采用懒加载；
本测试只覆盖 stdlib 即可验证的纯逻辑，不触发重依赖导入。
"""
import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.collect_local_882 import (  # noqa: E402
    ENGINE_PRESETS,
    build_base_url,
    build_generator_config,
    build_parser,
    resolve_target_config,
)


class TestEnginePresets(unittest.TestCase):
    def test_required_keys(self):
        for name, preset in ENGINE_PRESETS.items():
            for key in ("target_model", "api_model", "host", "port"):
                self.assertIn(key, preset, f"{name} 缺 {key}")

    def test_ports_match_plan(self):
        self.assertEqual(ENGINE_PRESETS["splash-qwen38"]["port"], 18080)
        self.assertEqual(ENGINE_PRESETS["splash-qwen36"]["port"], 18081)
        self.assertEqual(ENGINE_PRESETS["mlxvlm-gemma"]["port"], 18091)

    def test_model_ids(self):
        self.assertEqual(ENGINE_PRESETS["splash-qwen38"]["api_model"], "Qwen/Qwen3.8-27B")
        self.assertEqual(
            ENGINE_PRESETS["splash-qwen36"]["api_model"],
            "mlx-community/Qwen3.6-35B-A3B-4bit")
        self.assertEqual(
            ENGINE_PRESETS["mlxvlm-gemma"]["api_model"],
            "/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit")


class TestBuildBaseUrl(unittest.TestCase):
    def test_default(self):
        self.assertEqual(build_base_url("127.0.0.1", 18081), "http://127.0.0.1:18081/v1")

    def test_custom_host(self):
        self.assertEqual(build_base_url("192.168.3.5", 18080), "http://192.168.3.5:18080/v1")


class TestBuildGeneratorConfig(unittest.TestCase):
    def test_local_defaults(self):
        cfg = build_generator_config(
            target_model="Qwen3.8-27B",
            api_model="Qwen/Qwen3.8-27B",
            base_url="http://127.0.0.1:18080/v1",
            api_key="ollama",
            timeout=1800,
        )
        self.assertEqual(cfg["generator_type"], "direct")
        self.assertEqual(cfg["name"], "Qwen3.8-27B")
        self.assertEqual(cfg["api_model_name"], "Qwen/Qwen3.8-27B")
        self.assertEqual(cfg["base_url"], "http://127.0.0.1:18080/v1")
        self.assertEqual(cfg["api_key"], "ollama")  # 占位值，不写真实云凭据
        self.assertEqual(cfg["timeout"], 1800)
        # 端侧成本 = 0
        self.assertEqual(cfg["pricing"]["prompt_price_per_million"], 0.0)
        self.assertEqual(cfg["pricing"]["completion_price_per_million"], 0.0)
        self.assertEqual(cfg["temperature"], 0.0)
        self.assertEqual(cfg["top_p"], 1.0)

    def test_extra_body_defaults_to_empty(self):
        cfg = build_generator_config("m", "a", "http://x/v1", "ollama")
        self.assertEqual(cfg["extra_body"], {})


class TestArgparseDefaults(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args(["--engine", "splash-qwen36"])
        self.assertEqual(args.engine, "splash-qwen36")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.concurrency, 1)      # 本地 queue-size 1，单并发
        self.assertEqual(args.timeout, 1800)        # 本地推理慢，长超时
        self.assertEqual(args.api_key, "ollama")
        self.assertEqual(args.limit, 0)
        self.assertEqual(args.datasets, "")

    def test_unknown_engine_rejected(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--engine", "no-such-engine"])


class TestResolveTargetConfig(unittest.TestCase):
    def test_preset(self):
        args = build_parser().parse_args(["--engine", "splash-qwen38"])
        out = resolve_target_config(args)
        self.assertEqual(out["target_model"], "Qwen3.8-27B")
        self.assertEqual(out["api_model"], "Qwen/Qwen3.8-27B")
        self.assertEqual(out["base_url"], "http://127.0.0.1:18080/v1")

    def test_preset_override(self):
        args = build_parser().parse_args(
            ["--engine", "splash-qwen38", "--port", "9999", "--target-model", "my-model"])
        out = resolve_target_config(args)
        self.assertEqual(out["target_model"], "my-model")
        self.assertEqual(out["base_url"], "http://127.0.0.1:9999/v1")

    def test_custom_requires_fields(self):
        args = build_parser().parse_args(["--engine", "custom"])
        with self.assertRaises(SystemExit):
            resolve_target_config(args)

    def test_custom_with_fields(self):
        args = build_parser().parse_args([
            "--engine", "custom",
            "--api-model", "x/y",
            "--target-model", "local-model",
            "--port", "18081",
        ])
        out = resolve_target_config(args)
        self.assertEqual(out["target_model"], "local-model")
        self.assertEqual(out["api_model"], "x/y")
        self.assertEqual(out["base_url"], "http://127.0.0.1:18081/v1")

    def test_base_url_override_wins(self):
        args = build_parser().parse_args([
            "--engine", "splash-qwen36", "--base-url", "http://192.168.3.5:18081/v1"])
        out = resolve_target_config(args)
        self.assertEqual(out["base_url"], "http://192.168.3.5:18081/v1")


if __name__ == "__main__":
    unittest.main()
