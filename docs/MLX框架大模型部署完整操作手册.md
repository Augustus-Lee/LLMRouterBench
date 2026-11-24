# MLX 框架大模型部署完整操作手册

> 目标机器：`192.168.3.5` / `Jiuwen-MacMini` / Apple M5 Pro / 64GB
> 适用场景：在 Apple Silicon Mac 上通过 MLX 部署大语言模型，并进行单模型性能评测
> 已验证模型：Qwen3.6-35B-A3B-4bit、Gemma-4-26B-A4B-4bit
> 已验证运行时：Python 3.14.8、MLX 0.32.4、mlx-vlm 0.8.0、Transformers 5.24.0
> 文档更新时间：2026-11-24（Asia/Shanghai）

---

## 1. 文档目标与结论

本文给出从零到可评测的完整部署流程：

```text
检查机器
  → 创建 MLX 独立环境
  → 下载 MLX 模型
  → 校验模型完整性
  → 单次加载验证
  → 启动 OpenAI 兼容服务
  → 确认模型真正就绪
  → 发起推理请求
  → 记录性能指标
  → 停止服务并释放内存
```

本机已验证结果：

| 模型 | 本地目录 | 权重体积 | 加载峰值内存 | 状态 |
|---|---|---:|---:|---|
| Qwen3.6-35B-A3B-4bit | `/Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit` | 约 19.00GiB | 约 20.52GB | 已验证 |
| Gemma-4-26B-A4B-4bit | `/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit` | 约 14.29GiB | 约 15.42GB | 已验证 |

两个模型都通过：

- `mlx_vlm.generate` 单次生成；
- `mlx_vlm.server` 常驻加载；
- `/health` 的 `model_ready=true`；
- `/v1/models` 模型发现；
- `/v1/chat/completions` Chat 请求。

> 当前没有常驻模型服务。验证结束后服务均已停止。

---

## 2. 为什么三个文本目标模型使用 mlx-vlm

虽然 Qwen3.6-35B-A3B 和 Gemma-4-26B-A4B 在本轮只做文本输入，但它们不是普通纯文本 `*ForCausalLM`：

| 模型 | architecture | model_type | 运行时 |
|---|---|---|---|
| Qwen3.6-35B-A3B | `Qwen3_5MoeForConditionalGeneration` | `qwen3_5_moe` | `mlx-vlm` |
| Gemma-4-26B-A4B | `Gemma4ForConditionalGeneration` | `gemma4` | `mlx-vlm` |

因此：

- 即使输入只有文本，也使用 `mlx-vlm` 的 text-only 路径；
- 不用 `mlx-lm` 的结果替代 `mlx-vlm` 的性能结果；
- 三个文本目标模型使用相同运行时版本和同一套测试脚本；
- 后续 Qwen-Image 20B 属于扩散生图模型，不应与这三个文本模型混用 `mlx-vlm` 评测口径。

---

## 3. 部署前检查

### 3.1 登录机器

```bash
ssh jiuwen@192.168.3.5
```

### 3.2 检查芯片、内存与系统

```bash
hostname
sw_vers
sysctl -n machdep.cpu.brand_string
sysctl -n hw.memsize
system_profiler SPHardwareDataType
```

64GB 的实际字节数为：

```text
68719476736 bytes = 64GiB
```

### 3.3 检查磁盘空间

```bash
df -h /
du -sh ~/Models/* 2>/dev/null
```

建议至少保留 30GB 以上空闲空间，用于：

- 模型权重；
- Hugging Face / ModelScope 缓存；
- 评测日志；
- 结果数据和临时文件。

### 3.4 检查残留模型进程

```bash
pgrep -af 'mlx_vlm.server|mlx_lm.server|ollama|oMLX|EXO' || true
lsof -nP -iTCP:18090 -sTCP:LISTEN || true
lsof -nP -iTCP:18091 -sTCP:LISTEN || true
```

评测前应停止其他大模型服务。即使物理内存足够，并发模型也会改变 GPU、内存带宽和散热状态。

### 3.5 检查内存与 swap

```bash
memory_pressure
sysctl -n vm.swapusage
vm_stat
```

注意：

- swap 中有历史占用，不一定表示当前模型正在 swap；
- 正式测试前应记录初始 swap；
- 测试中若 swap 持续明显增长，应判定稳定性下降；
- 长时间未重启的机器可能积累较多缓存和历史 swap，正式基准前可安排在维护窗口重启。

本机 2026-11-24 检查时：

```text
total = 3072.00M  used = 1897.38M
uptime = 10 days
```

这不是当前模型 OOM，但说明正式测量前应记录并尽量恢复干净的机器状态。

### 3.6 检查系统负载

```bash
uptime
ps -Ao pid,%cpu,%mem,rss,comm -r | head -20
```

建议正式测试前 1/5/15 分钟负载稳定，且没有大型编译、视频转码、下载或模型推理任务。

---

## 4. 创建独立 MLX 环境

不要直接改动 Homebrew 的全局 Python 环境，避免升级 MLX 或 Transformers 后破坏其他服务。

### 4.1 创建虚拟环境

```bash
/opt/homebrew/bin/python3.14 -m venv ~/mlx-bench-venv
~/mlx-bench-venv/bin/python -m pip install --upgrade pip setuptools wheel
```

### 4.2 安装 mlx-vlm

```bash
~/mlx-bench-venv/bin/python -m pip install 'mlx-vlm==0.8.0'
```

若默认 PyPI 较慢：

```bash
~/mlx-bench-venv/bin/python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  'mlx-vlm==0.8.0'
```

### 4.3 安装模型下载工具

```bash
~/mlx-bench-venv/bin/python -m pip install 'modelscope==1.40.3'
```

### 4.4 验证安装版本

```bash
~/mlx-bench-venv/bin/python - <<'PY'
from importlib.metadata import version
for name in ("mlx", "mlx-vlm", "transformers", "modelscope"):
    print(name, version(name))
PY
```

本机已验证版本：

```text
mlx 0.32.4
mlx-vlm 0.8.0
transformers 5.24.0
modelscope 1.40.3
```

### 4.5 锁定环境

```bash
~/mlx-bench-venv/bin/python -m pip freeze \
  > ~/mlx-bench-venv/requirements-lock.txt
sha256sum ~/mlx-bench-venv/requirements-lock.txt 2>/dev/null || \
  shasum -a 256 ~/mlx-bench-venv/requirements-lock.txt
```

正式报告必须记录版本锁文件内容或哈希，否则结果不可复现。

---

## 5. 下载 MLX 模型

### 5.1 目录规范

```text
~/Models/Qwen3.6-35B-A3B-4bit
~/Models/gemma-4-26b-a4b-it-4bit
```

### 5.2 使用 ModelScope 下载

```bash
mkdir -p ~/Models

~/mlx-bench-venv/bin/modelscope download \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --local_dir ~/Models/Qwen3.6-35B-A3B-4bit
```

完成第一个模型后再下载第二个，不要并行下载：

```bash
~/mlx-bench-venv/bin/modelscope download \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --local_dir ~/Models/gemma-4-26b-a4b-it-4bit
```

### 5.3 断点续传

下载中断时：

1. 不要删除整个模型目录；
2. 保留已下载的 safetensors；
3. 重新执行完全相同的下载命令；
4. 下载完成后必须重新校验文件数量和总字节数。

不完整文件、Git LFS pointer 和下载中的临时文件都不能用于正式评测。

---

## 6. 校验模型完整性

### 6.1 检查 safetensors 数量和字节数

```bash
for model_dir in \
  ~/Models/Qwen3.6-35B-A3B-4bit \
  ~/Models/gemma-4-26b-a4b-it-4bit
do
  echo "=== $model_dir"
  find "$model_dir" -maxdepth 1 -name '*.safetensors' \
    -exec stat -f '%z %N' {} \; |
    awk '{sum += $1; count += 1}
         END {printf "safetensors=%.2f GiB files=%d\n", sum/1073741824, count}'
done
```

本机结果：

```text
Qwen3.6-35B-A3B-4bit  safetensors=19.00 GiB files=4
gemma-4-26b-a4b-it-4bit safetensors=14.29 GiB files=3
```

### 6.2 检查必要配置

```bash
for model_dir in \
  ~/Models/Qwen3.6-35B-A3B-4bit \
  ~/Models/gemma-4-26b-a4b-it-4bit
do
  echo "=== $model_dir"
  for name in config.json tokenizer.json tokenizer_config.json \
              processor_config.json model.safetensors.index.json; do
    test -s "$model_dir/$name" && echo "OK $name" || echo "MISSING $name"
  done
done
```

### 6.3 检查架构与上下文元数据

```bash
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path

for path in (
    Path.home() / "Models/Qwen3.6-35B-A3B-4bit",
    Path.home() / "Models/gemma-4-26b-a4b-it-4bit",
):
    cfg = json.loads((path / "config.json").read_text())
    text = cfg.get("text_config", {})
    print(path.name)
    print("  model_type:", cfg.get("model_type"))
    print("  architectures:", cfg.get("architectures"))
    print("  max_position_embeddings:", text.get("max_position_embeddings"))
    print("  sliding_window:", text.get("sliding_window"))
PY
```

已确认：

```text
Qwen3.6-35B-A3B-4bit    qwen3_5_moe   max_position_embeddings=262144
gemma-4-26b-a4b-it-4bit gemma4        max_position_embeddings=262144
```

`262144` 是模型元数据支持值，不代表当前实际可稳定运行 256K 上下文。正式结果必须逐级实测。

### 6.4 检查 safetensors 是否可被 MLX 正常加载

最直接的校验是执行第 7 节的单次生成。文件存在但结构损坏、index 不匹配或权重缺失，会在加载阶段报错。

---

## 7. 单次 CLI 加载验证

### 7.1 Qwen3.6-35B-A3B

```bash
~/mlx-bench-venv/bin/mlx_vlm.generate \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --prompt "只回答：Qwen3.6 加载成功" \
  --max-tokens 32 \
  --temperature 0.0 \
  --no-verbose
```

### 7.2 Gemma-4-26B-A4B

```bash
~/mlx-bench-venv/bin/mlx_vlm.generate \
  --model ~/Models/gemma-4-26b-a4b-it-4bit \
  --prompt "只回答：Gemma-4 加载成功" \
  --max-tokens 32 \
  --temperature 0.0 \
  --no-verbose
```

### 7.3 判定标准

成功必须同时满足：

- 进程正常结束；
- 有非空文本输出；
- 没有 OOM、Metal error、缺失权重或 tokenizer 错误。

这条命令适合验证环境和模型，不适合做正式吞吐量评测，因为：

- 每次请求都会重新加载模型；
- 单次结果受冷启动和首次 Metal 图构建影响；
- 文本没有经过统一 Chat 模板；
- 不能代表持续并发性能。

---

## 8. 启动 OpenAI 兼容服务

持续评测推荐 `mlx_vlm.server`。它让模型加载一次，然后通过 OpenAI Chat Completions 接口接收多个请求。

### 8.1 创建日志与 PID 目录

```bash
mkdir -p ~/llm-deploy/{logs,run}
```

### 8.2 启动 Qwen3.6

```bash
nohup ~/mlx-bench-venv/bin/mlx_vlm.server \
  --host 127.0.0.1 \
  --port 18090 \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --max-tokens 2048 \
  --log-level INFO \
  > ~/llm-deploy/logs/qwen36-server.log 2>&1 < /dev/null &

echo $! > ~/llm-deploy/run/qwen36-server.pid
```

### 8.3 启动 Gemma-4

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

### 8.4 单模型原则

不要让两个模型同时常驻。

虽然 64GB 可能能同时装下约 19GB 和约 15GB 的权重，但同时加载会：

- 压缩 KV Cache 和系统余量；
- 改变内存带宽与 GPU 调度；
- 让 Prefill/Decode 和峰值内存结果不可比；
- 增加 swap 与热降频风险。

严格执行：

```text
加载模型 A → 完成测试 → 停止并确认内存释放 → 加载模型 B
```

### 8.5 可选的安全与资源参数

局域网访问时启用 API Key：

```bash
export MLX_VLM_SERVER_API_KEY='<由密钥管理器生成的长随机值>'

~/mlx-bench-venv/bin/mlx_vlm.server \
  --host 0.0.0.0 \
  --port 18090 \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --api-key "$MLX_VLM_SERVER_API_KEY"
```

不要：

- 在脚本、文档或 Git 中写真实密钥；
- 把无鉴权的服务直接暴露到公网；
- 在共享网络中默认监听 `0.0.0.0`。

如果不确定网络是否可信，使用第 12 节的 SSH 隧道。

长上下文场景可按需限制 KV Cache：

```bash
--max-kv-size 32768
```

但这会改变可测试上下文和性能口径。正式对比时，各模型必须使用相同设置，并记录参数。

---

## 9. 判断服务是否真正就绪

### 9.1 不能只看端口

启动后 `/health` 可能很快返回 HTTP 200，但模型仍在后台逐层加载。必须检查：

```json
{
  "model_ready": true,
  "preload_in_progress": false,
  "error": null
}
```

### 9.2 等待 Qwen3.6 就绪

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

### 9.3 等待 Gemma-4 就绪

把地址改成：

```text
http://127.0.0.1:18091/health
```

### 9.4 常见状态解释

| `/health` 状态 | 含义 | 操作 |
|---|---|---|
| `preload_in_progress=true` | 模型正在逐层加载 | 继续等待 |
| `model_ready=false` | 尚未可推理 | 不要开始评测 |
| `model_ready=true` | 可以推理 | 进行 `/v1/models` 和 Chat 验证 |
| `error` 非空 | 加载出错 | 查看服务日志 |
| HTTP 503 | 服务存在但模型未就绪 | 通常不是模型加载失败 |

查看日志：

```bash
tail -f ~/llm-deploy/logs/qwen36-server.log
tail -f ~/llm-deploy/logs/gemma4-server.log
```

---

## 10. 验证 OpenAI 接口

### 10.1 查看模型 ID

Qwen3.6：

```bash
curl -fsS http://127.0.0.1:18090/v1/models |
  /usr/bin/python3 -m json.tool
```

Gemma-4：

```bash
curl -fsS http://127.0.0.1:18091/v1/models |
  /usr/bin/python3 -m json.tool
```

返回的模型 ID 是绝对路径：

```text
/Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit
/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit
```

### 10.2 Qwen3.6 Chat 请求

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

### 10.3 Gemma-4 Chat 请求

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

### 10.4 响应中的性能字段

正常响应包含：

- `choices[0].message.content`
- `usage.prompt_tokens`
- `usage.completion_tokens`
- `timings.prompt_per_second`
- `timings.predicted_per_second`
- `timings.peak_memory`

本机冒烟结果：

| 模型 | 输出 | 峰值内存 |
|---|---|---:|
| Qwen3.6-35B-A3B | 服务加载成功 | 约 20.52GB |
| Gemma-4-26B-A4B | 服务加载成功 | 约 15.42GB |

这些是接口冒烟值，不是正式性能基准。

---

## 11. Python API 加载与自定义计时

### 11.1 基础调用

```python
from mlx_vlm import load, generate

MODEL_PATH = "/Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit"

model, processor = load(MODEL_PATH)

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
```

Gemma-4 只需替换：

```python
MODEL_PATH = "/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit"
```

`GenerationResult` 常用字段：

| 字段 | 含义 |
|---|---|
| `text` | 生成文本 |
| `prompt_tokens` | 输入 token 数 |
| `generation_tokens` | 输出 token 数 |
| `prompt_tps` | Prefill token/s |
| `generation_tps` | Decode token/s |
| `peak_memory` | MLX 报告峰值内存，单位 GB |
| `finish_reason` | 停止原因 |
| `token_ids` | 生成 token ID |

### 11.2 计时脚本要求

正式性能脚本应做到：

1. 一个进程只加载一个模型；
2. 先做至少 1 次 warm-up，丢弃结果；
3. 固定 prompt、输入 token 数和 `max_tokens`；
4. 固定 `temperature=0`、`top_p`、`seed`；
5. 每个场景至少重复 3 次；
6. 同时记录 MLX 峰值、进程 RSS 和 swap；
7. 测完退出进程，再加载下一个模型。

不要用裸 `generate()` 的单次输出替代完整 Chat 模板路径。对话模型评测应统一使用 `mlx_vlm.server` 或统一构造相同 Chat 模板。

---

## 12. 接入 LLMRouterBench

### 12.1 直接从部署机器运行

如果 LLMRouterBench 和 MLX 服务在同一台 Mac 上：

```yaml
models:
  - name: qwen36-35b-a3b-mlx
    api_model_name: /Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit
    base_url: http://127.0.0.1:18090/v1
    api_key: local-no-auth
    temperature: 0.0
    top_p: 1.0
    timeout: 1800
    generator_type: direct
```

Gemma-4：

```yaml
models:
  - name: gemma4-26b-a4b-mlx
    api_model_name: /Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit
    base_url: http://127.0.0.1:18091/v1
    api_key: local-no-auth
    temperature: 0.0
    top_p: 1.0
    timeout: 1800
    generator_type: direct
```

说明：

- `name` 是结果目录和缓存使用的名称，不能包含 `/`；
- `api_model_name` 必须与 `/v1/models` 返回的 ID 一致；
- 服务未启用鉴权时，`api_key` 可以使用非空占位字符串，OpenAI 客户端通常不允许空值；
- 性能测试应设置 `concurrency: 1`；
- 正式生产密钥必须从环境变量或密钥管理器读取，不写入配置。

### 12.2 从另一台机器通过 SSH 隧道接入

推荐使用 SSH 本地端口转发，不直接开放远端端口：

```bash
ssh -N \
  -L 18090:127.0.0.1:18090 \
  jiuwen@192.168.3.5
```

然后本机配置：

```yaml
base_url: http://127.0.0.1:18090/v1
api_model_name: /Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit
```

Gemma-4 再建立另一条隧道：

```bash
ssh -N \
  -L 18091:127.0.0.1:18091 \
  jiuwen@192.168.3.5
```

注意：

- 服务监听 `127.0.0.1` 时，SSH 隧道仍可访问；
- 不需要把 MLX 服务改为 `0.0.0.0`；
- 不要让两个模型同时常驻，隧道可以同时存在，但应只启动当前要测的那一个服务。

### 12.3 OpenAI Python 客户端验证

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:18090/v1",
    api_key="local-no-auth",
)

response = client.chat.completions.create(
    model="/Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit",
    messages=[
        {"role": "user", "content": "只回答：客户端调用成功"}
    ],
    max_tokens=32,
    temperature=0,
)

print(response.choices[0].message.content)
print(response.usage)
```

> 如果运行环境没有安装 `openai`，先在运行 LLMRouterBench 的环境中安装，而不是修改 MLX 推理环境。

---

## 13. 性能指标与记录口径

### 13.1 必测指标

| 指标 | 说明 |
|---|---|
| 冷加载时间 | 进程启动到 `model_ready=true` |
| TTFT | 请求开始到首个输出 token |
| Prefill token/s | 输入 token 数 / Prefill 时间 |
| Decode token/s | 输出 token 数 / Decode 时间 |
| MLX 峰值内存 | `peak_memory` 或 `mx.get_peak_memory()` |
| 进程 RSS / footprint | 定时采样进程内存 |
| swap | 测试前后及测试中的 swap 变化 |
| 成功/失败 | OOM、超时、Metal 错误、进程退出 |

### 13.2 固定参数

```text
temperature = 0
top_p = 1.0
max_tokens = 256（主）/ 512（复核）
batch_size = 1
concurrency = 1
seed = 42
```

不启用：

- speculative decoding；
- draft model；
- prompt cache；
- 不同模型间不一致的 KV Cache 量化。

### 13.3 上下文阶梯

```text
128 → 1K → 4K → 8K → 16K → 32K → 64K → 128K
```

每个场景：

```text
warm-up 1 次（丢弃）
+ 正式测量 3 次
+ 记录 median / min / max
```

判定最大可用上下文必须同时满足：

- 请求完成；
- 输出有效；
- 无 OOM；
- swap 无显著增长；
- 进程未崩溃；
- 延迟没有数量级异常。

### 13.4 结果元数据

每条结果至少记录：

```text
hostname
chip / CPU / GPU 核数
unified_memory_gb
macOS_version
python_version
mlx_version
mlx_vlm_version
transformers_version
model_repo
model_revision
quantization
safetensors_count
safetensors_bytes
prompt_tokens
max_tokens
temperature
top_p
seed
ttft_ms
prefill_tokens_per_sec
decode_tokens_per_sec
peak_memory_gb
process_rss_mb
swap_before_mb
swap_after_mb
status
```

---

## 14. 停止服务与释放内存

### 14.1 使用 PID 文件停止

Qwen3.6：

```bash
kill "$(cat ~/llm-deploy/run/qwen36-server.pid)"
```

Gemma-4：

```bash
kill "$(cat ~/llm-deploy/run/gemma4-server.pid)"
```

### 14.2 等待并确认退出

```bash
sleep 2
pgrep -af mlx_vlm.server || true
lsof -nP -iTCP:18090 -sTCP:LISTEN || true
lsof -nP -iTCP:18091 -sTCP:LISTEN || true
```

### 14.3 检查内存回收

```bash
ps -Ao pid,rss,comm -r | head -10
memory_pressure
sysctl -n vm.swapusage
```

如果进程已退出但系统内存没有立即回落：

- 不代表模型仍在运行；
- macOS 文件缓存和页面回收可能延迟；
- 以 `pgrep` 无残留进程和端口无监听为准；
- 正式开始下一模型前等待 15～30 秒，并确认 swap 未持续增长。

---

## 15. 常见故障排查

### 15.1 `/health` 可访问但 Chat 返回 503

原因：

```text
模型仍在预加载
```

处理：

```bash
curl -fsS http://127.0.0.1:18090/health |
  /usr/bin/python3 -m json.tool
```

等待 `model_ready=true`，不要重启服务。

### 15.2 `/v1/models` 返回空列表

通常是模型尚未就绪。先检查 `/health`，再检查日志：

```bash
tail -100 ~/llm-deploy/logs/qwen36-server.log
```

### 15.3 端口被占用

```bash
lsof -nP -iTCP:18090 -sTCP:LISTEN
lsof -nP -iTCP:18091 -sTCP:LISTEN
```

不要直接使用默认 `8080`，本机已有其他服务占用记录。评测端口使用 `18090` / `18091`。

### 15.4 模型加载慢或内存压力高

检查：

```bash
vm_stat
sysctl -n vm.swapusage
memory_pressure
pgrep -af 'mlx_vlm|ollama|oMLX|EXO' || true
```

处理：

- 停止其他模型服务；
- 关闭大型编译、转码和下载任务；
- 降低 `--max-tokens` 或 `--max-kv-size`；
- 缩小测试上下文；
- 重启机器后重测，不要在同一脏状态下比较数字。

### 15.5 Gemma 出现 audio mel filter 警告

本机加载 Gemma-4 时曾出现 `transformers/audio_utils.py` 的 mel filter 警告，但文本生成成功。若文本请求正常，该警告不是故障；若出现异常，再检查模型卡与 Transformers 兼容性。

### 15.6 Python 导入 MLX 版本失败

不要使用：

```python
import mlx
print(mlx.__version__)
```

新版 MLX 可能不导出 `__version__`。改用：

```python
from importlib.metadata import version
print(version("mlx"))
```

### 15.7 下载超时或中断

处理顺序：

1. 保留已有文件；
2. 检查网络；
3. 重新执行相同 ModelScope 下载命令；
4. 下载完成后校验 safetensors 数量和总字节数；
5. 再执行单次加载验证。

不要使用只有几百字节的 Git LFS pointer 冒充完整权重。

---

## 16. 单模型部署检查清单

### 环境

- [ ] 确认机器为 Apple Silicon；
- [ ] 确认 macOS 和磁盘空间；
- [ ] 创建 `~/mlx-bench-venv`；
- [ ] 安装并锁定 `mlx-vlm` 版本；
- [ ] 记录 Python、MLX、Transformers 版本。

### 模型

- [ ] 下载完整 MLX 权重；
- [ ] 校验 safetensors 数量和字节数；
- [ ] 校验 config、tokenizer、processor 和 index；
- [ ] 确认 model_type 与 architecture；
- [ ] 记录模型路径和 revision。

### 服务

- [ ] 只启动一个模型；
- [ ] 端口为 18090 或 18091；
- [ ] `/health` 的 `model_ready=true`；
- [ ] `/v1/models` 可见模型；
- [ ] Chat Completions 返回有效文本；
- [ ] 记录峰值内存和初始 swap。

### 测试

- [ ] warm-up 1 次并丢弃；
- [ ] 固定输入、输出和采样参数；
- [ ] 每个场景重复至少 3 次；
- [ ] 记录 Prefill、Decode、TTFT；
- [ ] 记录 MLX 峰值、RSS、swap；
- [ ] 异常时整组复跑。

### 收尾

- [ ] 停止服务；
- [ ] 确认无残留进程；
- [ ] 确认端口无监听；
- [ ] 保存日志、结果和版本锁文件；
- [ ] 等待内存和温度恢复后再测下一模型。

---

## 17. 最短命令清单

### Qwen3.6

```bash
ssh jiuwen@192.168.3.5

mkdir -p ~/llm-deploy/{logs,run}

nohup ~/mlx-bench-venv/bin/mlx_vlm.server \
  --host 127.0.0.1 \
  --port 18090 \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --max-tokens 2048 \
  > ~/llm-deploy/logs/qwen36-server.log 2>&1 < /dev/null &

echo $! > ~/llm-deploy/run/qwen36-server.pid

# 等待 model_ready=true
curl -fsS http://127.0.0.1:18090/health |
  /usr/bin/python3 -m json.tool

# 验证可用
curl -fsS http://127.0.0.1:18090/v1/models |
  /usr/bin/python3 -m json.tool

# 测试后停止
kill "$(cat ~/llm-deploy/run/qwen36-server.pid)"
```

### Gemma-4

```bash
ssh jiuwen@192.168.3.5

nohup ~/mlx-bench-venv/bin/mlx_vlm.server \
  --host 127.0.0.1 \
  --port 18091 \
  --model ~/Models/gemma-4-26b-a4b-it-4bit \
  --max-tokens 2048 \
  > ~/llm-deploy/logs/gemma4-server.log 2>&1 < /dev/null &

echo $! > ~/llm-deploy/run/gemma4-server.pid

# 等待 model_ready=true
curl -fsS http://127.0.0.1:18091/health |
  /usr/bin/python3 -m json.tool

# 测试后停止
kill "$(cat ~/llm-deploy/run/gemma4-server.pid)"
```

---

## 18. 相关文档

- [MLX 模型加载与 OpenAI 兼容服务操作手册](./MLX_模型加载与OpenAI服务操作手册.md)
- [MLX 单模型性能评测：选机、部署与实施方案](./MLX_单模型性能评测_选机与实施方案.md)
- [MLX 文本模型与 Qwen-Image 生图模型运行时规划](./MLX_文本与生图模型运行时规划.md)
- [单模型性能评测耗时评估与机器独占通知](./单模型性能评测耗时评估与机器独占通知.md)
