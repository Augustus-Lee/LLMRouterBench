# -*- coding: utf-8 -*-
"""Mac 文本模型性能评测公共库。

对应 docs/MAC_文本模型性能评测方案.md（v4，Splash + MLX-VLM 分引擎口径）5.5-5.8 节：
- 定长 prompt 合成（上下文阶梯 1K/8K/32K/64K/128K）；
- OpenAI 兼容客户端（Splash / MLX-VLM 共用），流式测 TTFT、非流式取服务端计时；
- 重复次数统计（均值/中位数/P95，最近邻秩法）；
- 结果 JSON schema（缺失字段记 null，不补 0、不估算）；
- macOS 系统指标解析器（swap / wired / iostat）与可选采样器。

仅依赖标准库；psutil/matplotlib 缺失时相关能力优雅降级为 None/跳过。
"""
from __future__ import annotations

import json
import hashlib
import datetime
import math
import os
import platform
import random
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# Splash Qwen3.8=18080 / Qwen3.6=18081；MLX-VLM Gemma=18091（方案 1.3 节端口约定）
DEFAULT_CONTEXT_LADDER: Tuple[int, ...] = (1024, 8192, 32768, 65536, 131072)

_VOCAB: Tuple[str, ...] = (
    "the", "of", "and", "in", "to", "a", "is", "that", "for", "it",
    "model", "tokens", "context", "memory", "latency", "throughput",
    "attention", "cache", "layer", "expert", "routing", "benchmark",
    "prefill", "decode", "sequence", "window", "engine", "quantized",
    "inference", "weights", "batch", "stream", "request", "response",
    "server", "client", "timeout", "sample", "repeat", "measure",
)


class BenchRequestError(RuntimeError):
    """OpenAI 兼容请求失败（HTTP 错误、连接失败、超时、响应不完整）。"""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# prompt 合成与 token 估算
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：英文按词计 1，CJK 按字计 1。

    仅用于合成 prompt 的长度控制；正式记录中的 token 数以服务端 usage 为准。
    """
    if not text:
        return 0
    total = 0
    for word in text.split():
        cjk = len(_CJK_RE.findall(word))
        rest = _CJK_RE.sub("", word)
        total += cjk + (1 if rest else 0)
    return total


def synthesize_prompt(target_tokens: int, seed: int = 42) -> str:
    """合成约 target_tokens 个 token 的英文 prompt（词数即 token 数）。

    确定性：同一 (target, seed) 输出完全一致，保证各档位可复现对比。
    纯英文词表使得 estimate_tokens == 词数，长度控制精确无累计误差。
    """
    if target_tokens <= 0:
        return ""
    rng = random.Random(seed)
    words: List[str] = []
    total = 0
    while total < target_tokens:
        n = rng.randint(8, 14)
        take = min(n, target_tokens - total)
        words.extend(rng.choice(_VOCAB) for _ in range(take))
        total += take
    return " ".join(words)


def load_workload_records(path: str, contexts: Optional[Sequence[int]] = None
                           ) -> List[Dict[str, Any]]:
    """读取构建器生成的 performance JSONL，并按目标 token 档位过滤。

    文件是逐模型 tokenizer 校准的；``contexts`` 仍是目标档位，不是某个
    服务端实际返回的 prompt_tokens。读取阶段只做结构校验，服务端 usage
    仍是正式测量中的最终 token 数。
    """
    wanted = set(contexts) if contexts is not None else None
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{line_no}: record is not an object")
            for key in ("id", "layer", "target_tokens", "actual_tokens", "text"):
                if rec.get(key) in (None, ""):
                    raise ValueError(f"{path}:{line_no}: missing {key}")
            if not isinstance(rec["target_tokens"], int):
                raise ValueError(f"{path}:{line_no}: target_tokens must be int")
            if not isinstance(rec["actual_tokens"], int):
                raise ValueError(f"{path}:{line_no}: actual_tokens must be int")
            if wanted is not None and rec["target_tokens"] not in wanted:
                continue
            records.append(rec)
    if not records:
        raise ValueError(f"{path}: no workload records selected by contexts {sorted(wanted or [])}")
    return records


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def summarize_repeats(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """对重复测量值计算均值/中位数/P95（最近邻秩法）/最差值。

    None 值过滤掉（不补 0）；空列表返回全 None。
    """
    clean = sorted(v for v in values if v is not None)
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None,
                "min": None, "max": None}
    idx = max(0, math.ceil(0.95 * n) - 1)
    return {
        "n": n,
        "mean": sum(clean) / n,
        "median": statistics.median(clean),
        "p95": clean[idx],
        "min": clean[0],
        "max": clean[-1],
    }


# ---------------------------------------------------------------------------
# 结果 JSON schema（方案 8.1 节）
# ---------------------------------------------------------------------------

def make_run_record(
    engine: str,
    engine_version: str,
    model: str,
    port: int,
    context_tokens: int,
    max_output_tokens: int,
    draft_model: Optional[str] = None,
    kv_format: Optional[str] = None,
    state_format: Optional[str] = None,
    weight_format: Optional[str] = None,
    quantization: Optional[str] = None,
    hardware: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    cold_load_s: Optional[float] = None,
    warmup_runs: int = 1,
    repeat_runs: int = 3,
    revision_or_digest: Optional[str] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """构造单档上下文的运行记录；未测字段一律 None，不补 0。"""
    if hardware is None:
        hardware = detect_hardware()
    return {
        "engine": engine,
        "engine_version": engine_version,
        "port": port,
        "model": model,
        "model_revision_or_digest": revision_or_digest,
        "weight_format": weight_format,
        "quantization": quantization,
        "draft_model": draft_model,
        "kv_format": kv_format,
        "state_format": state_format,
        "hardware": hardware,
        "context_tokens": context_tokens,
        "prompt_target_tokens": context_tokens,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "cold_load_s": cold_load_s,
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
        "repeats": [],
        "memory": {
            "peak_process_gib": None,
            "wired_gib": None,
            "swap_delta_gib": None,
            "memory_pressure": None,
        },
        "ssd_read_mib_s": None,
        "error_count": 0,
        "success": True,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# macOS 系统指标解析器（纯函数，可单测）
# ---------------------------------------------------------------------------

_SWAP_RE = re.compile(
    r"total:\s*([\d.]+)\s*([MG])i?B?\s+"
    r"used:\s*([\d.]+)\s*([MG])i?B?\s+"
    r"free:\s*([\d.]+)\s*([MG])i?B?",
    re.IGNORECASE,
)


def _to_mib(value: float, unit: str) -> float:
    return value * 1024.0 if unit.upper() == "G" else value


def parse_swapusage(text: str) -> Optional[Dict[str, float]]:
    """解析 `sysctl vm.swapusage` 输出，单位统一为 MiB。"""
    m = _SWAP_RE.search(text)
    if not m:
        return None
    return {
        "total_mib": _to_mib(float(m.group(1)), m.group(2)),
        "used_mib": _to_mib(float(m.group(3)), m.group(4)),
        "free_mib": _to_mib(float(m.group(5)), m.group(6)),
    }


def parse_vm_stat_wired_pages(text: str) -> Optional[int]:
    """解析 `vm_stat` 输出中的 wired 页数。"""
    m = re.search(r"Pages wired down:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def parse_vm_stat_page_size(text: str) -> Optional[int]:
    """解析 `vm_stat` 头部的页大小（Apple Silicon 通常 16384）。"""
    m = re.search(r"page size of\s+(\d+)\s+bytes", text)
    return int(m.group(1)) if m else None


def swap_delta_mib(before: Optional[Dict[str, float]],
                   after: Optional[Dict[str, float]]) -> Optional[float]:
    """两次 swapusage 读数之间的 used 增量（MiB）；任一缺失返回 None。"""
    if before is None or after is None:
        return None
    return after["used_mib"] - before["used_mib"]


def parse_iostat_mib_per_s(text: str) -> Optional[float]:
    """解析 macOS `iostat -d` 数据行的 MB/s 列（最后一行数值列）。"""
    value = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                candidate = float(parts[-1])
            except ValueError:
                continue
            value = candidate
    return value


# ---------------------------------------------------------------------------
# 服务端计时字段提取
# ---------------------------------------------------------------------------

_MS_TIMING_KEYS = ("prompt_ms", "prompt_tps", "prompt_generation_ms")
_GEN_MS_KEYS = ("generation_ms", "predicted_ms", "decode_ms", "predicted_per_token_ms")


def extract_server_timings(data: Dict[str, Any]) -> Tuple[
        Optional[float], Optional[float], Optional[str]]:
    """从响应 JSON 提取服务端 (prefill_s, decode_s, source)。

    兼容两种风格，字段缺失返回 None（不补 0）：
    - llama.cpp/Splash 风格：timings.prompt_ms / timings.generation_ms（毫秒）；
    - Ollama 风格：prompt_eval_duration / eval_duration（纳秒）。
    """
    timings = data.get("timings")
    if isinstance(timings, dict):
        prompt_ms = timings.get("prompt_ms")
        gen_ms = None
        for key in _GEN_MS_KEYS:
            if timings.get(key) is not None:
                gen_ms = timings[key]
                break
        if prompt_ms is not None and gen_ms is not None:
            try:
                return float(prompt_ms) / 1000.0, float(gen_ms) / 1000.0, "server"
            except (TypeError, ValueError):
                return None, None, None
    prompt_ns = data.get("prompt_eval_duration")
    eval_ns = data.get("eval_duration")
    if prompt_ns is not None and eval_ns is not None:
        try:
            return float(prompt_ns) / 1e9, float(eval_ns) / 1e9, "server"
        except (TypeError, ValueError):
            return None, None, None
    return None, None, None


# ---------------------------------------------------------------------------
# OpenAI 兼容客户端（Splash / MLX-VLM 共用）
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _extract_server_wall_s(data: Dict[str, Any]) -> Optional[float]:
    """从 response 提取服务端总处理时长（秒）。"""
    rl = data.get("request_latency")
    if isinstance(rl, dict) and rl.get("wall_ms") is not None:
        try:
            return float(rl["wall_ms"]) / 1000.0
        except (TypeError, ValueError):
            return None
    timings = data.get("timings")
    if isinstance(timings, dict) and timings.get("wall_ms") is not None:
        try:
            return float(timings["wall_ms"]) / 1000.0
        except (TypeError, ValueError):
            return None
    return None


def _extract_server_ttft(data: Dict[str, Any]) -> Optional[float]:
    """从响应提取服务端 TTFT（秒）。

    兼容多种格式（优先级从高到低）：
      - Splash: metrics.request_latency.ttft_ms
      - Splash fallback: metrics.request_latency.start_to_first_token_ms
      - 顶层 request_latency.ttft_ms
      - timings.ttft_ms
    """
    rl = (data.get("metrics") or {}).get("request_latency")
    if isinstance(rl, dict):
        for key in ("ttft_ms", "start_to_first_token_ms"):
            if rl.get(key) is not None:
                try:
                    return float(rl[key]) / 1000.0
                except (TypeError, ValueError):
                    return None
    rl_top = data.get("request_latency")
    if isinstance(rl_top, dict) and rl_top.get("ttft_ms") is not None:
        try:
            return float(rl_top["ttft_ms"]) / 1000.0
        except (TypeError, ValueError):
            return None
    timings = data.get("timings")
    if isinstance(timings, dict) and timings.get("ttft_ms") is not None:
        try:
            return float(timings["ttft_ms"]) / 1000.0
        except (TypeError, ValueError):
            return None
    return None


def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_reasoning_tokens(data: Dict[str, Any],
                              usage: Optional[Dict[str, Any]]) -> int:
    """reasoning token 数；服务端不提供时返回 -1，不使用 0 冒充。"""
    candidates = []
    if isinstance(usage, dict):
        candidates.append(usage.get("reasoning_tokens"))
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            candidates.append(details.get("reasoning_tokens"))
    candidates.append(data.get("reasoning_tokens"))
    value = _first_int(*candidates)
    return value if value is not None else -1


def _extract_cache_status(usage: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[int]]:
    """提取 prompt 缓存命中状态与 token 数。

    未返回缓存字段时保持 None；0 命中显式标记为 miss。
    """
    if not isinstance(usage, dict):
        return None, None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = _first_int(details.get("cached_tokens"), details.get("cache_hit_tokens"))
    else:
        cached = _first_int(usage.get("cached_tokens"), usage.get("cache_hit_tokens"))
    if cached is None:
        return None, None
    prompt_tokens = _first_int(usage.get("prompt_tokens"))
    if cached == 0:
        return "miss", 0
    if prompt_tokens is not None:
        if cached >= prompt_tokens:
            return "hit", cached
        return "partial", cached
    return "hit", cached


class OpenAICompatClient:
    """最小 OpenAI /v1/chat/completions 客户端（stdlib urllib 实现）。

    - 非流式：服务端 usage + timings 字段优先，缺失记 None；
    - 流式：客户端时间戳测 TTFT / decode 区间，usage 缺失时按内容 chunk 计数。
    """

    def __init__(self, host: str, port: int, model: str,
                 timeout_s: float = 1800.0) -> None:
        self.host = host
        self.port = port
        self.model = model
        self.timeout_s = timeout_s
        self.url = f"http://{host}:{port}/v1/chat/completions"

    # -- 基础 ---------------------------------------------------------------

    def _post(self, payload: Dict[str, Any]) -> Any:
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:  # 4xx/5xx
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise BenchRequestError(f"HTTP {exc.code}: {detail}", status_code=exc.code) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise BenchRequestError(f"request failed: {exc}") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise BenchRequestError(f"invalid JSON response: {exc}") from exc

    @staticmethod
    def _messages(messages: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
        if messages:
            return messages
        return [{"role": "user", "content": "ping"}]

    # -- 非流式 ---------------------------------------------------------------

    def chat_nonstream(self, messages: Optional[List[Dict[str, str]]] = None,
                       max_tokens: int = 256, temperature: float = 0.0,
                       top_p: float = 1.0, seed: int = 42) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": self._messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": False,
        }
        request_start_time = _iso_now()
        started = time.perf_counter()
        data = self._post(payload)
        response_end_time = _iso_now()
        e2e_s = time.perf_counter() - started

        choices = data.get("choices") or []
        finish_reason = choices[0].get("finish_reason") if choices else None
        content = None
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        reasoning_tokens = _extract_reasoning_tokens(data, usage)
        cache_status, cache_hit_tokens = _extract_cache_status(usage)
        server_wall_s = _extract_server_wall_s(data)
        prefill_s, decode_s, source = extract_server_timings(data)

        prefill_tps = None
        if prefill_s and prefill_s > 0 and prompt_tokens:
            prefill_tps = prompt_tokens / prefill_s
        decode_tps = None
        if decode_s and decode_s > 0 and completion_tokens:
            decode_tps = completion_tokens / decode_s

        return {
            "stream": False,
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_status": cache_status,
            "cache_hit_tokens": cache_hit_tokens,
            "server_wall_s": server_wall_s,
            "request_start_time": request_start_time,
            "first_token_time": None,
            "response_end_time": response_end_time,
            "ttft_s": _extract_server_ttft(data),
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "prefill_tok_per_s": prefill_tps,
            "decode_tok_per_s": decode_tps,
            "e2e_s": e2e_s,
            "timing_source": source,
            "finish_reason": finish_reason,
            "error": None,
            "success": True,
        }

    # -- 流式 ---------------------------------------------------------------

    def chat_stream(self, messages: Optional[List[Dict[str, str]]] = None,
                    max_tokens: int = 256, temperature: float = 0.0,
                    top_p: float = 1.0, seed: int = 42) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": self._messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": True,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request_start_time = _iso_now()
        started = time.perf_counter()
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise BenchRequestError(f"HTTP {exc.code}: {detail}", status_code=exc.code) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise BenchRequestError(f"request failed: {exc}") from exc

        ttft_s: Optional[float] = None
        first_content_t: Optional[float] = None
        last_chunk_t: Optional[float] = None
        content_chunks = 0
        content_parts: List[str] = []
        finish_reason: Optional[str] = None
        usage: Optional[Dict[str, Any]] = None
        first_token_time: Optional[str] = None

        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:"):].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except ValueError:
                    continue
                now = time.perf_counter()
                last_chunk_t = now
                choices = chunk.get("choices") or []
                if choices:
                    finish = choices[0].get("finish_reason")
                    if finish:
                        finish_reason = finish
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        content_chunks += 1
                        content_parts.append(delta["content"])
                        if first_content_t is None:
                            first_content_t = now
                            ttft_s = now - started
                            first_token_time = _iso_now()
                if chunk.get("usage"):
                    usage = chunk["usage"]
        except (OSError, urllib.error.URLError) as exc:
            raise BenchRequestError(f"stream interrupted: {exc}") from exc
        finally:
            resp.close()

        response_end_time = _iso_now()
        e2e_s = (last_chunk_t or time.perf_counter()) - started
        prompt_tokens = usage.get("prompt_tokens") if usage else None
        if usage and usage.get("completion_tokens") is not None:
            completion_tokens = usage.get("completion_tokens")
        else:
            completion_tokens = content_chunks if content_chunks else None
        reasoning_tokens = _extract_reasoning_tokens({}, usage)
        cache_status, cache_hit_tokens = _extract_cache_status(usage)

        decode_s: Optional[float] = None
        if first_content_t is not None and last_chunk_t is not None:
            decode_s = max(0.0, last_chunk_t - first_content_t)
        decode_tps = None
        if decode_s and decode_s > 0 and completion_tokens:
            decode_tps = completion_tokens / decode_s

        return {
            "stream": True,
            "content": "".join(content_parts) if content_parts else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_status": cache_status,
            "cache_hit_tokens": cache_hit_tokens,
            "server_wall_s": None,
            "request_start_time": request_start_time,
            "first_token_time": first_token_time,
            "response_end_time": response_end_time,
            "ttft_s": ttft_s,
            "prefill_s": None,  # 客户端无法拆分 prefill
            "decode_s": decode_s,
            "prefill_tok_per_s": None,
            "decode_tok_per_s": decode_tps,
            "e2e_s": e2e_s,
            "timing_source": "client",
            "finish_reason": finish_reason,
            "error": None,
            "success": True,
        }


# ---------------------------------------------------------------------------
# 系统指标采样器（可选依赖降级）
# ---------------------------------------------------------------------------

def _run_cmd(cmd: List[str], timeout: float = 10.0) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
        if out.returncode == 0:
            return out.stdout
        return None
    except (OSError, subprocess.SubprocessError):
        return None


def detect_hardware() -> Dict[str, Any]:
    """采集本机硬件 fingerprint（macOS sysctl，失败降级为 unknown）。"""
    def sysctl(name: str) -> Optional[str]:
        out = _run_cmd(["sysctl", "-n", name])
        return out.strip() if out else None

    is_mac = platform.system() == "Darwin"
    memsize = sysctl("hw.memsize") if is_mac else None
    try:
        mem_gib = int(memsize) / (1024 ** 3) if memsize else None
    except ValueError:
        mem_gib = None
    return {
        "host": socket.gethostname() or "unknown",
        "chip": (sysctl("machdep.cpu.brand_string") if is_mac else None) or "unknown",
        "model_identifier": (sysctl("hw.model") if is_mac else None) or "unknown",
        "memory_gib": round(mem_gib, 1) if mem_gib else None,
        "os_version": platform.mac_ver()[0] if is_mac else platform.platform(),
    }


class SystemMonitor:
    """运行期间采样：进程峰值 RSS（psutil 可选）/ wired / swap 增量 / 磁盘速率。

    所有能力独立降级：psutil 缺失 → RSS 为 None；非 macOS → 系统指标为 None；
    未指定 disk → 不采样 iostat。stop() 返回的键与结果 schema 对应。
    """

    def __init__(self, pid: Optional[int] = None, sample_interval: float = 0.5,
                 disk: Optional[str] = None) -> None:
        self.pid = pid
        self.sample_interval = sample_interval
        self.disk = disk
        self._peak_rss = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._iostat_values: List[float] = []
        self._swap_before: Optional[Dict[str, float]] = None
        self._psutil = None

    # -- 内部 ---------------------------------------------------------------

    def _read_swap(self) -> Optional[Dict[str, float]]:
        out = _run_cmd(["sysctl", "-n", "vm.swapusage"])
        return parse_swapusage(out) if out else None

    def _read_wired_pages(self) -> Optional[int]:
        out = _run_cmd(["vm_stat"])
        return parse_vm_stat_wired_pages(out) if out else None

    def _sample_loop(self) -> None:
        if self._psutil is not None and self.pid:
            try:
                proc = self._psutil.Process(self.pid)
            except Exception:
                proc = None
            while not self._stop_event.is_set():
                if proc is not None:
                    try:
                        self._peak_rss = max(self._peak_rss,
                                             proc.memory_info().rss)
                    except Exception:
                        pass
                self._stop_event.wait(self.sample_interval)

    def _iostat_loop(self) -> None:
        while not self._stop_event.is_set():
            out = _run_cmd(["iostat", "-c", "2", "-w", "1", "-d", self.disk or "disk0"])
            if out:
                value = parse_iostat_mib_per_s(out)
                if value is not None:
                    self._iostat_values.append(value)
            self._stop_event.wait(1.0)

    # -- 生命周期 ---------------------------------------------------------------

    def start(self) -> None:
        try:
            import psutil  # noqa: WPS433 可选依赖
            self._psutil = psutil
        except ImportError:
            self._psutil = None
        if platform.system() == "Darwin":
            self._swap_before = self._read_swap()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        if self.disk and platform.system() == "Darwin":
            t = threading.Thread(target=self._iostat_loop, daemon=True)
            t.start()

    def stop(self) -> Tuple[Dict[str, Any], Optional[float]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        memory = {
            "peak_process_gib": None,
            "wired_gib": None,
            "swap_delta_gib": None,
            "memory_pressure": None,
        }
        if self._peak_rss:
            memory["peak_process_gib"] = round(self._peak_rss / (1024 ** 3), 3)
        if platform.system() == "Darwin":
            stat_out = _run_cmd(["vm_stat"])
            if stat_out:
                pages = parse_vm_stat_wired_pages(stat_out)
                page_size = parse_vm_stat_page_size(stat_out) or 16384
                if pages is not None:
                    memory["wired_gib"] = round(pages * page_size / (1024 ** 3), 3)
            swap_after = self._read_swap()
            delta = swap_delta_mib(self._swap_before, swap_after)
            if delta is not None:
                memory["swap_delta_gib"] = round(delta / 1024.0, 4)
            mp_out = _run_cmd(["memory_pressure", "-Q"])
            if mp_out:
                memory["memory_pressure"] = mp_out.strip()[:200]
        ssd_read = None
        if self._iostat_values:
            ssd_read = round(sum(self._iostat_values) / len(self._iostat_values), 2)
        return memory, ssd_read


# ---------------------------------------------------------------------------
# 评测主循环（适配器共用）
# ---------------------------------------------------------------------------

def run_one_request(client: OpenAICompatClient, messages: List[Dict[str, str]],
                    max_tokens: int, temperature: float, top_p: float,
                    seed: int, repeat_index: int, stream: bool) -> Dict[str, Any]:
    """单次请求封装：异常捕获为 error 记录，不中断整个矩阵。"""
    try:
        if stream:
            entry = client.chat_stream(messages=messages, max_tokens=max_tokens,
                                       temperature=temperature, top_p=top_p,
                                       seed=seed)
        else:
            entry = client.chat_nonstream(messages=messages, max_tokens=max_tokens,
                                          temperature=temperature, top_p=top_p,
                                          seed=seed)
        entry["repeat"] = repeat_index
        return entry
    except BenchRequestError as exc:
        return {
            "stream": stream,
            "repeat": repeat_index,
            "prompt_tokens": None,
            "completion_tokens": None,
            "ttft_s": None,
            "prefill_s": None,
            "decode_s": None,
            "prefill_tok_per_s": None,
            "decode_tok_per_s": None,
            "e2e_s": None,
            "timing_source": None,
            "finish_reason": None,
            "error": str(exc),
            "success": False,
        }


def run_benchmark(args: Any, client: OpenAICompatClient) -> List[str]:
    """按上下文阶梯执行完整矩阵，写出每档一个 JSON 记录。

    每档流程（方案 5.7 节）：warm-up ≥1（不计入正式数据）→
    正式 repeat 次，每次同时跑非流式与流式两种口径 →
    并行采样 RSS/swap/wired/iostat → 写 JSON。
    """
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    contexts: List[int] = args.contexts
    workload_file = getattr(args, "workload_file", None)
    workload_file_sha256: Optional[str] = None

    if workload_file:
        workloads = load_workload_records(workload_file, contexts=contexts)
        workload_file_sha256 = file_sha256(workload_file)
        print(f"[bench] workload={workload_file} "
              f"selected={len(workloads)} sha256={workload_file_sha256[:12]}")
    else:
        workloads = [{
            "id": None,
            "layer": "synthetic",
            "target_tokens": ctx,
            "actual_tokens": ctx,
            "text": synthesize_prompt(ctx, seed=args.seed),
            "text_sha256": None,
        } for ctx in contexts]

    for item in workloads:
        # context_tokens 是测试档位；actual_tokens 只作为 workload 构建时
        # 的 tokenizer 校准值。正式输入长度仍以服务端 prompt_tokens 为准。
        ctx = int(item["target_tokens"])
        record = make_run_record(
            engine=args.engine,
            engine_version=args.engine_version,
            model=args.model,
            port=args.port,
            context_tokens=ctx,
            max_output_tokens=args.max_output_tokens,
            draft_model=getattr(args, "draft_model", None),
            kv_format=getattr(args, "kv_format", None),
            state_format=getattr(args, "state_format", None),
            weight_format=getattr(args, "weight_format", None),
            quantization=getattr(args, "quantization", None),
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            cold_load_s=args.cold_load_s,
            warmup_runs=args.warmup,
            repeat_runs=args.repeat,
            revision_or_digest=getattr(args, "revision_or_digest", None),
        )
        prompt = str(item["text"])
        workload_id = item.get("id")
        if workload_id:
            record.update({
                "workload_id": workload_id,
                "workload_file": workload_file,
                "workload_file_sha256": workload_file_sha256,
                "workload_sha256": item.get("text_sha256"),
                "workload_layer": item.get("layer"),
                "workload_target_tokens": item.get("target_tokens"),
                "workload_calibrated_tokens": item.get("actual_tokens"),
                "workload_tokenizer": item.get("tokenizer"),
                "workload_tokenizer_revision": item.get("tokenizer_revision"),
                "workload_tokenizer_exact": item.get("tokenizer_exact"),
                "source_dataset": item.get("source_dataset"),
                "source_revision": item.get("source_revision"),
                "source_split": item.get("source_split"),
                "source_ids": item.get("source_ids"),
                "source_license": item.get("source_license"),
                "construction": item.get("construction"),
                "boundary_probe": bool(item.get("boundary_probe")),
                "question_included": bool(item.get("question_included")),
            })
        else:
            record.update({
                "workload_id": None,
                "workload_file": None,
                "workload_file_sha256": None,
                "workload_sha256": None,
                "workload_layer": "synthetic",
                "workload_target_tokens": ctx,
                "workload_calibrated_tokens": ctx,
                "workload_tokenizer": None,
                "workload_tokenizer_revision": None,
                "workload_tokenizer_exact": False,
                "source_dataset": None,
                "source_revision": None,
                "source_split": None,
                "source_ids": [],
                "source_license": None,
                "construction": "synthetic_repeated_vocabulary",
                "boundary_probe": ctx >= 262144,
                "question_included": False,
            })
        messages = [{"role": "user", "content": prompt}]

        monitor = SystemMonitor(pid=getattr(args, "server_pid", None),
                                disk=getattr(args, "monitor_disk", None))
        monitor.start()

        notes: List[str] = []
        for w in range(args.warmup):
            try:
                client.chat_nonstream(messages=messages,
                                      max_tokens=args.max_output_tokens,
                                      temperature=args.temperature,
                                      top_p=args.top_p, seed=args.seed)
            except BenchRequestError as exc:
                notes.append(f"warmup[{w}] failed: {exc}")

        for i in range(args.repeat):
            for stream_mode in (False, True):
                entry = run_one_request(client, messages, args.max_output_tokens,
                                        args.temperature, args.top_p, args.seed,
                                        i, stream_mode)
                record["repeats"].append(entry)

        memory, ssd_read = monitor.stop()
        record["memory"] = memory
        record["ssd_read_mib_s"] = ssd_read
        record["error_count"] = sum(1 for e in record["repeats"] if not e["success"])
        record["success"] = any(e["success"] for e in record["repeats"])
        if notes:
            record["notes"] = "; ".join(notes)[:500]

        if workload_id:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(workload_id))
            filename = f"{args.engine}_{safe_id}.json"
        else:
            filename = f"{args.engine}_{ctx}.json"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        written.append(path)
        print(f"[bench] ctx={ctx} workload={workload_id or 'synthetic'} -> {path} "
              f"(ok={record['success']}, errors={record['error_count']})")
    return written
