# -*- coding: utf-8 -*-
"""OpenAICompatClient 集成测试：本地 mock OpenAI 兼容服务（含 SSE 流式）。"""
import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.bench_common import OpenAICompatClient, BenchRequestError  # noqa: E402


class MockHandler(BaseHTTPRequestHandler):
    """可配置行为的 mock：类属性 mode 控制。"""

    mode = "ok"          # ok | no_timings | error500
    chunk_delay = 0.05
    chunk_count = 5

    def log_message(self, *args):  # 静音
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/chat/completions") is False:
            self.send_error(404)
            return
        if MockHandler.mode == "error500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"internal error")
            return
        if body.get("stream"):
            self._serve_stream()
        else:
            self._serve_nonstream()

    def _serve_nonstream(self):
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 32},
        }
        if MockHandler.mode != "no_timings":
            payload["timings"] = {"prompt_ms": 1000.0, "generation_ms": 4000.0}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(MockHandler.chunk_delay)  # 模拟 prefill 延迟
        # 首个内容 chunk
        first = {"choices": [{"delta": {"role": "assistant", "content": "a"}, "finish_reason": None}]}
        self.wfile.write(b"data: " + json.dumps(first).encode() + b"\n\n")
        self.wfile.flush()
        time.sleep(MockHandler.chunk_delay)
        for _ in range(MockHandler.chunk_count - 1):
            chunk = {"choices": [{"delta": {"content": "b"}, "finish_reason": None}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(MockHandler.chunk_delay)
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        if MockHandler.mode != "no_usage":
            final["usage"] = {"prompt_tokens": 100, "completion_tokens": 42}
        self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class ClientTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()


class TestNonStream(ClientTestBase):
    def test_parses_usage_and_server_timings(self):
        client = OpenAICompatClient(host="127.0.0.1", port=self.port, model="mock")
        out = client.chat_nonstream(max_tokens=32)
        self.assertTrue(out["success"])
        self.assertEqual(out["prompt_tokens"], 100)
        self.assertEqual(out["completion_tokens"], 32)
        self.assertAlmostEqual(out["prefill_s"], 1.0)
        self.assertAlmostEqual(out["decode_s"], 4.0)
        self.assertEqual(out["timing_source"], "server")
        # tok/s 从服务端 timing 派生
        self.assertAlmostEqual(out["prefill_tok_per_s"], 100.0)
        self.assertAlmostEqual(out["decode_tok_per_s"], 8.0)
        self.assertEqual(out["finish_reason"], "stop")

    def test_missing_timings_are_none_not_zero(self):
        MockHandler.mode = "no_timings"
        try:
            client = OpenAICompatClient(host="127.0.0.1", port=self.port, model="mock")
            out = client.chat_nonstream(max_tokens=32)
            self.assertIsNone(out["prefill_s"])
            self.assertIsNone(out["decode_s"])
            self.assertIsNone(out["prefill_tok_per_s"])
            self.assertIsNone(out["decode_tok_per_s"])
            self.assertIsNone(out["timing_source"])
        finally:
            MockHandler.mode = "ok"

    def test_http_500_raises(self):
        MockHandler.mode = "error500"
        try:
            client = OpenAICompatClient(host="127.0.0.1", port=self.port, model="mock")
            with self.assertRaises(BenchRequestError):
                client.chat_nonstream(max_tokens=32)
        finally:
            MockHandler.mode = "ok"


class TestStream(ClientTestBase):
    def test_ttft_measured_client_side(self):
        MockHandler.chunk_delay = 0.15
        try:
            client = OpenAICompatClient(host="127.0.0.1", port=self.port, model="mock")
            out = client.chat_stream(max_tokens=64)
            self.assertTrue(out["success"])
            self.assertGreaterEqual(out["ttft_s"], 0.05)
            self.assertEqual(out["timing_source"], "client")
            self.assertEqual(out["finish_reason"], "stop")
            # usage 优先于 chunk 计数
            self.assertEqual(out["completion_tokens"], 42)
            # decode 区间 = 首内容 chunk 之后的时间
            self.assertGreater(out["decode_s"], 0.0)
            self.assertGreater(out["decode_tok_per_s"], 0.0)
            self.assertGreaterEqual(out["e2e_s"], out["ttft_s"])
        finally:
            MockHandler.chunk_delay = 0.05

    def test_stream_without_usage_counts_chunks(self):
        MockHandler.mode = "no_usage"
        MockHandler.chunk_count = 4
        try:
            client = OpenAICompatClient(host="127.0.0.1", port=self.port, model="mock")
            out = client.chat_stream(max_tokens=64)
            # 无 usage 时按内容 chunk 计数
            self.assertEqual(out["completion_tokens"], 4)
            self.assertEqual(out["timing_source"], "client")
        finally:
            MockHandler.mode = "ok"
            MockHandler.chunk_count = 5


if __name__ == "__main__":
    unittest.main()
