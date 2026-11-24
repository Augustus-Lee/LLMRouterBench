# MLX 模型加载与 OpenAI 兼容服务操作手册

> 目标机器：`192.168.3.5` / `macmini-m5-jiuwen` / Apple M5 Pro / 64GB
> 运行环境：`/Users/jiuwen/mlx-bench-venv`
> 已验证版本：Python 3.14.8、mlx 0.32.4、mlx-vlm 0.8.0、transformers 5.24.0
> 验证日期：2026-11-23（Asia/Shanghai）

---

## 1. 结论先行

两个模型都已经下载完成，并且已经实测可以通过 `mlx-vlm` 加载和推理：

| 模型 | 本地目录 | 权重体积 | 冒烟加载峰值内存 |
|---|---|---:|---:|
| Qwen3.6-35B-A3B-4bit | `/Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit` | 约 19.00GiB | 约 20.52GB |
| Gemma-4-26B-A4B-4bit | `/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit` | 约 14.29GiB | 约 15.42GB |

这里的“load”不是单独执行一个“挂载”命令，而是：

1. `mlx-vlm` 读取模型配置、tokenizer 和 4bit 权重；
2. 将权重加载到 Apple Silicon 统一内存；
3. 第一次生成时建立 Metal 执行图并分配 KV Cache、激活值和临时缓冲；
4. 模型才进入可稳定推理状态。

推荐方式：

| 目的 | 推荐方式 |
|---|---|
| 快速检查模型是否可加载 | `mlx_vlm.generate` 单次命令 |
| 给评测脚本或 LLMRouterBench 调用 | `mlx_vlm.server` 常驻 OpenAI 兼容服务 |
| 自定义 Prefill/Decode 计时 | Python API：`load()` + `generate()` |

> 当前没有常驻服务。冒烟验证完成后，两个服务均已停止，没有残留 `mlx_vlm.server` 进程。

---

## 2. 方式一：单次 CLI 加载验证

这种方式会加载模型、生成一次回答，然后退出。适合确认模型、环境、Metal 和 tokenizer 是否正常。

### 2.1 加载 Qwen3.6-35B-A3B

```bash
ssh jiuwen@192.168.3.5

~/mlx-bench-venv/bin/mlx_vlm.generate \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --prompt "只回答：Qwen3.6 加载成功" \
  --max-tokens 32 \
  --temperature 0.0 \
  --no-verbose
```

### 2.2 加载 Gemma-4-26B-A4B

```bash
ssh jiuwen@192.168.3.5

~/mlx-bench-venv/bin/mlx_vlm.generate \
  --model ~/Models/gemma-4-26b-a4b-it-4bit \
  --prompt "只回答：Gemma-4 加载成功" \
  --max-tokens 32 \
  --temperature 0.0 \
  --no-verbose
```

如果命令最终输出回答文本且进程正常退出，说明模型加载成功。

注意：

- 命令执行期间会占用约 15～21GB 统一内存，应避免同时运行另一个大模型；
- 每次命令都会重新加载模型，不能用于持续吞吐量评测；
- `--max-tokens` 是单次最多生成 token 数，不是模型上下文窗口；
- 仅用这条命令得到的一次耗时，不等于正式 Prefill/Decode 基准结果。

---

## 3. 方式二：常驻 OpenAI 兼容服务

这是后续接入 LLMRouterBench、采集器或自定义压测脚本的推荐方式。

### 3.1 启动 Qwen3.6 服务

在 `192.168.3.5` 上执行：

```bash
mkdir -p ~/llm-deploy/{logs,run}

nohup ~/mlx-bench-venv/bin/mlx_vlm.server \
  --host 127.0.0.1 \
  --port 18090 \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --max-tokens 2048 \
  --log-level INFO \
  > ~/llm-deploy/logs/qwen36-server.log 2>&1 < /dev/null &

echo $! > ~/llm-deploy/run/qwen36-server.pid
```

Gemma-4 使用另一个端口：

```bash
nohup ~/mlx-bench-venv/bin/mlx_vlm.server \
  --host 127.0.0.1 \
  --port 18091 \
  --model ~/Models/gemma-4-26b-a4b-it-4bit \
  --max-tokens 2048 \
  --log-level INFO \
  > ~/llm-deploy/logs/gemma4-server.log 2>&1 < /dev/null &

echo $! > ~/llm-deploy/run/gemma4-server.pid
```

> `18090` 和 `18091` 是分开的端口，但不要让两个模型同时常驻。64GB 机器虽然可能容纳两份权重，仍会显著压缩 KV Cache 和系统余量，并污染单模型性能结果。必须遵循“加载一个、测完停止、再加载另一个”。

### 3.2 判断模型是否真正加载完成

服务启动后，HTTP 接口可能很快就能访问，但此时模型仍在后台预加载。因此不能只判断“端口能连”，必须等待 `/health` 返回：

```json
{
  "model_ready": true,
  "preload_in_progress": false,
  "error": null
}
```

等待 Qwen3.6 就绪：

```bash
for i in $(seq 1 180); do
  health="$(curl -fsS http://127.0.0.1:18090/health 2>/dev/null || true)"
  printf '%s\n' "$health"
  if printf '%s' "$health" | /usr/bin/python3 -c \
    'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("model_ready") is True else 1)' \
    2>/dev/null; then
    echo "Qwen3.6 ready"
    break
  fi
  sleep 1
done
```

等待 Gemma-4 就绪时，只需把地址改成：

```text
http://127.0.0.1:18091/health
```

如果 `/health` 显示 `preload_in_progress: true`，说明仍在逐层加载；此时 `/v1/chat/completions` 返回 503 属于正常现象，不是模型加载失败。

### 3.3 查看模型 ID

```bash
curl -fsS http://127.0.0.1:18090/v1/models | /usr/bin/python3 -m json.tool
```

Qwen3.6 返回的模型 ID 是本地绝对路径：

```text
/Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit
```

Gemma-4 的模型 ID 是：

```text
/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit
```

### 3.4 验证 Chat Completions

Qwen3.6：

```bash
curl -fsS http://127.0.0.1:18090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit",
    "messages": [
      {"role": "user", "content": "只回答：服务加载成功"}
    ],
    "max_tokens": 32,
    "temperature": 0
  }' | /usr/bin/python3 -m json.tool
```

Gemma-4：

```bash
curl -fsS http://127.0.0.1:18091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit",
    "messages": [
      {"role": "user", "content": "只回答：服务加载成功"}
    ],
    "max_tokens": 32,
    "temperature": 0
  }' | /usr/bin/python3 -m json.tool
```

正常响应包含：

- `choices[0].message.content`
- `usage.prompt_tokens`
- `usage.completion_tokens`
- `timings.prompt_per_second`
- `timings.predicted_per_second`
- `timings.peak_memory`

这些字段可以直接用于性能采集，但正式结果仍应固定 prompt、上下文长度、生成长度和重复次数。

### 3.5 停止服务

优先使用 PID 文件，避免误杀另一个模型：

```bash
kill "$(cat ~/llm-deploy/run/qwen36-server.pid)"
```

Gemma-4：

```bash
kill "$(cat ~/llm-deploy/run/gemma4-server.pid)"
```

确认没有残留：

```bash
pgrep -af mlx_vlm.server || true
lsof -nP -iTCP:18090 -sTCP:LISTEN || true
lsof -nP -iTCP:18091 -sTCP:LISTEN || true
```

---

## 4. 方式三：Python API 加载

适合需要自定义计时、收集 MLX 指标或批量构造请求的场景。

```bash
~/mlx-bench-venv/bin/python - <<'PY'
from mlx_vlm import load, generate

model_path = "/Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit"

model, processor = load(model_path)

result = generate(
    model,
    processor,
    prompt="只回答：Python API 加载成功",
    max_tokens=32,
    temperature=0.0,
    verbose=False,
)

print("text:", result.text)
print("prompt_tokens:", result.prompt_tokens)
print("generation_tokens:", result.generation_tokens)
print("prompt_tps:", result.prompt_tps)
print("generation_tps:", result.generation_tps)
print("peak_memory_gb:", result.peak_memory)
PY
```

Gemma-4 只需替换：

```python
model_path = "/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit"
```

`generate()` 返回 `GenerationResult`，常用字段如下：

| 字段 | 含义 |
|---|---|
| `text` | 生成文本 |
| `prompt_tokens` | 输入 token 数 |
| `generation_tokens` | 输出 token 数 |
| `prompt_tps` | Prefill token/s |
| `generation_tps` | Decode token/s |
| `peak_memory` | MLX 报告的峰值内存，单位 GB |
| `finish_reason` | 停止原因 |
| `token_ids` | 生成 token ID |

注意：

- 一个 Python 进程只加载一个模型；
- 测完一个模型后退出进程，再启动另一个模型，避免内存碎片和缓存残留；
- 如果使用裸 `generate()`，它不等同于 OpenAI Chat 模板路径；做对话模型评测时，应显式使用 chat 模板或统一走 `mlx_vlm.server`，保证输入格式一致；
- 第一次生成包含 Metal 图构建和缓存初始化，正式测速前至少做一次 warm-up，并将 warm-up 请求排除在统计之外。

---

## 5. 接入 LLMRouterBench 时的参数

如果评测脚本和服务运行在同一台 Mac mini 上：

```text
base_url = http://127.0.0.1:18090/v1
api_key = 任意非空占位值，或服务启用鉴权后使用真实值
model = /Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit
```

Gemma-4：

```text
base_url = http://127.0.0.1:18091/v1
model = /Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit
```

如果评测程序在另一台机器上运行，需要把服务监听地址改为：

```bash
--host 0.0.0.0
```

然后使用：

```text
http://192.168.3.5:18090/v1
```

但这会把服务暴露到局域网。仅在可信内网使用，并建议启用服务鉴权：

```bash
--api-key "$MLX_VLM_SERVER_API_KEY"
```

不要在文档、脚本或聊天记录中写真实密钥。

---

## 6. 已验证结果与使用边界

本次冒烟验证结果：

| 模型 | `/health` | `/v1/models` | Chat Completions | 峰值内存 |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | `model_ready=true` | 通过 | 通过 | 约 20.52GB |
| Gemma-4-26B-A4B | `model_ready=true` | 通过 | 通过 | 约 15.42GB |

验证结束后已停止两个服务，并确认：

```text
无 mlx_vlm.server 残留进程
18090 / 18091 无监听
```

这些数据只证明“加载链路和 OpenAI 接口工作正常”，不是最终性能基准。正式评测至少还需要：

1. 固定 prompt 集和输入长度；
2. 固定输出长度与采样参数；
3. 分别记录冷加载时间、TTFT、Prefill token/s、Decode token/s；
4. 记录 MLX 峰值内存、进程 RSS 和 swap；
5. 每个场景重复多次并报告中位数、均值、P95；
6. 每次只加载一个模型，测试后停止服务。

---

## 7. 最短操作清单

### Qwen3.6

```bash
ssh jiuwen@192.168.3.5
mkdir -p ~/llm-deploy/{logs,run}
nohup ~/mlx-bench-venv/bin/mlx_vlm.server \
  --host 127.0.0.1 --port 18090 \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --max-tokens 2048 \
  > ~/llm-deploy/logs/qwen36-server.log 2>&1 &
echo $! > ~/llm-deploy/run/qwen36-server.pid

# 等到 /health 的 model_ready 为 true
curl -fsS http://127.0.0.1:18090/health | /usr/bin/python3 -m json.tool

# 测完停止
kill "$(cat ~/llm-deploy/run/qwen36-server.pid)"
```

### Gemma-4

```bash
ssh jiuwen@192.168.3.5
nohup ~/mlx-bench-venv/bin/mlx_vlm.server \
  --host 127.0.0.1 --port 18091 \
  --model ~/Models/gemma-4-26b-a4b-it-4bit \
  --max-tokens 2048 \
  > ~/llm-deploy/logs/gemma4-server.log 2>&1 &
echo $! > ~/llm-deploy/run/gemma4-server.pid

# 等到 /health 的 model_ready 为 true
curl -fsS http://127.0.0.1:18091/health | /usr/bin/python3 -m json.tool

# 测完停止
kill "$(cat ~/llm-deploy/run/gemma4-server.pid)"
```
