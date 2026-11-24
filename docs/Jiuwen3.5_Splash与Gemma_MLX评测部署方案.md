# Jiuwen 3.5 Splash 与 Gemma MLX 评测部署方案

> 核查日期：2026-11-24（Asia/Shanghai）  
> 目标机器：`192.168.3.5` / `Jiuwen-MacMini` / SSH alias `macmini-m5-jiuwen`  
> 硬件：Apple M5 Pro / 18 CPU 核 / 20 GPU 核 / 64GB 统一内存 / macOS 27.2  
> 目标负载：单模型、单实例、文本性能评测  
> 框架划分：Qwen 系列使用 Splash 1.5.0；Gemma-4-26B-A4B 使用 MLX-VLM 0.8.0  
> 原则：不同推理引擎的结果必须分表记录，不能把 MLX、Splash、Ollama 的数字混在同一张对比表中。

---

## 1. 结论摘要

**可以在 jiuwen 3.5 上部署 Splash 并进行大模型性能测试，但需要满足资源独占条件。**

| 模型 | 推荐引擎 | 当前资产 | 可行性 | 结论 |
|---|---|---|---|---|
| Qwen3.8-27B | Splash 1.5.0 | GGUF Q4_K_M、mmproj、DFlash2 均完整 | ✅ 可立即做加载验证 | **首轮 Splash 基线首选** |
| Qwen3.6-35B-A3B | Splash 1.5.0 | 已有 19GB MLX 4bit 主模型；缺配套 DFlash2 | ⚠️ 补约 1.05GB draft 后可用 | **第二优先，无需重下主模型** |
| Gemma-4-26B-A4B | MLX-VLM 0.8.0 | 完整 14GB MLX 4bit 权重 | ✅ 已具备加载验证条件 | **按要求走 MLX，不走 Splash** |
| Qwen3.8-Flash-Next | Splash 1.5.0 | 当前无完整资产 | ⚠️ 仅可做低比特/SSD 流式实验 | **不作为本轮正式基线** |


---

## 2. 机器配置与当前运行状态

### 2.1 硬件与系统

| 项目 | 实测值 |
|---|---|
| 主机名 | `Jiuwen-MacMini` |
| 地址 | `192.168.3.5` |
| Model Identifier | `Mac17,16` |
| 芯片 | Apple M5 Pro |
| CPU | 18 核（6 个 Super 核 + 12 个 Performance 核） |
| GPU | 20 核，Metal 4 |
| 统一内存 | 64GB（68,719,476,736 bytes） |
| 系统版本 | macOS 27.2 |
| 数据盘 | 926GiB，已用约 432GiB，可用约 454GiB（49% 使用率） |

### 2.2 本次只读核查时正在运行的服务

| 服务 | 端口 | 资源占用 | 影响 |
|---|---:|---:|---|
| NVIDIA PAIR | `8080`、`11434` | 代理进程 | 本身不是主要模型负载 |
| Ollama | `127.0.0.1:11435` | `llama-server` RSS 约 30.2GiB | **显著占用统一内存和 GPU** |
| Ollama 已加载模型 | 内部端口 | `qwen3.6:35b-a3b-q4_K_M`，上下文 65,536 | 与正式测试目标重叠 |
| MLX-LM | `8090` | RSS 约 5.2GiB | 额外占用 Metal 和内存 |
| MLX-LM 已加载模型 | `8090` | `Qwen3.5-9B-4bit` | 与 Gemma/Qwen 测试冲突 |

相关进程特征：

```text
llama-server ... qwen3.6 ... -c 65536 -np 1
mlx_lm.server --model /Users/jiuwen/Models/Qwen3.5-9B-4bit --port 8090
```

### 2.3 正式测试的环境底线

正式结果至少应满足：

- 单模型、单实例；
- 单请求串行执行，不混入其他推理；
- 停止或错开 Ollama、MLX-LM 和 NVIDIA PAIR 模型负载；
- 固定输入长度、输出长度、采样参数、KV 精度和上下文上限；
- 固定模型 revision、量化版本、权重文件大小和启动命令；
- 记录冷加载时间、TTFT、Prefill tok/s、Decode tok/s、峰值 RSS、wired memory、swap、SSD 读取；
- 至少一次 warm-up，正式重复 3～5 次并报告均值、中位数、P95；
- 不将旧日志中的交互式请求当作正式基准；
- 不跨引擎混表。

> 未取得机器使用方同意前，不应直接停止 NVIDIA PAIR、Ollama 或 MLX-LM 服务。

---

## 3. Splash 1.5.0 能力与边界

### 3.1 安装状态

```text
/opt/homebrew/bin/splash -> ../Cellar/splash/1.5.0/bin/splash
Splash 1.5.0
```

无需重新安装。核查安装源码后确认，Splash 1.5.0 原生支持的模型族为：

```text
Qwen3.8-27B
Qwen3.6-35B-A3B
Qwen3.5-122B-A10B
Qwen3.8-Flash-Next
Flint
```

所覆盖的底层架构为：

```text
qwen35
qwen35moe
qwen4exp
flint
```

**不支持 `gemma4` / `gemma4_text`。**

### 3.2 Splash 的本地权重能力

Splash 1.5.0 的 `--model-dir` 同时支持：

- 本地 GGUF 文件或目录；
- 本地 MLX affine 权重目录；
- 本地 DFlash2 draft 目录；
- GGUF 的本地 mmproj。

因此，Qwen3.6 当前已有的 MLX 4bit 主权重可以直接作为 Splash 的本地目标模型，但还需要匹配的 DFlash2 geometry。只补 DFlash2，不需要重新下载 19GB 主权重。

### 3.3 已执行的只读兼容性检查

对现有 Qwen3.6 MLX 目录执行 Splash 文本架构检查，结果为：

```text
OK format=mlx-affine
family=Qwen3.6-35B-A3B
vision=none
```

对 Gemma-4-26B-A4B 执行同一检查，结果为：

```text
ModelError: no supported model has this architecture (text model type gemma4_text);
supported: Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B,
Qwen3.8-Flash-Next, Flint
```

这证明 Gemma 不是机器或格式问题，而是 Splash 架构不支持。

### 3.4 与评测相关的关键参数

| 参数 | 作用 | 对测量结果的影响 |
|---|---|---|
| `--max-memory` | Metal 内存预算 | 太低会降速/失败；太高会挤压系统 |
| `--max-context` | 上下文 token 上限 | 决定 KV、state 与缓存压力 |
| `--kv-format int8\|bf16` | KV Cache 精度 | INT8 更省内存；BF16 更接近质量上限 |
| `--state-format fp32\|bf16` | 循环状态精度 | 影响长上下文内存与数值稳定性 |
| `--draft-model` | DFlash2 投机解码 | 显著影响 Decode 速度与结果可复现性 |
| `--verify-rows all` | 校验全部 draft 行 | 可复现性更好，速度较慢 |
| `--verify-rows auto` | 自适应选择校验行 | 速度更快，MoE 贪心输出可能不完全可复现 |
| `--queue-size 1` | 单请求准入 | 防止并发污染 benchmark |
| `--decode-share` | 多请求时 prefill/decode 时间分配 | 单请求正式测试应固定并说明 |
| `--idle-release` | 空闲后释放权重 | 共享机器更安全，但复测会重新加载 |
| `--expert-memory` | MoE 专家 RAM 槽位 | 不足时会从 SSD 读专家，速度下降 |
| `--ple-memory` | Flash-Next 的 PLE/n-gram 表缓存 | 不设或过小会流式读 SSD |
| `--sparse-kv-memory` | Flash-Next 稀疏 KV 缓存 | 影响长上下文性能 |

### 3.5 三个容易混淆的“支持”

- **架构支持 ≠ 资产完整**：模型族在支持列表中，不代表本地已有正确 GGUF、MLX 权重和 DFlash2。
- **能加载 ≠ 能稳定正式评测**：加载成功不代表 128K、256K 不会 OOM 或 swap。
- **能出 tok/s ≠ 发挥最大性能**：量化、上下文、KV 精度、DFlash2、SSD 流式读取都会改变结果。

---

## 4. Qwen3.8-27B Splash 部署方案

### 4.1 当前资产

| 资产 | 路径 | 大小 | 状态 |
|---|---|---:|---|
| 主 GGUF | `/Users/jiuwen/Models/splash/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` | 16,464,440,224 bytes ≈ 15.33GiB | ✅ 完整 |
| 视觉投影 | `/Users/jiuwen/Models/splash/Qwen3.8-27B-GGUF/mmproj-BF16.gguf` | 931,146,432 bytes ≈ 887MiB | ✅ 完整 |
| DFlash2 | `/Users/jiuwen/Models/splash/Qwen3.8-27B-DFlash2/model.safetensors` | 3,848,817,896 bytes ≈ 3.58GiB | ✅ 完整 |

同目录其他 Q5/Q6/Q8 GGUF 多为 135～136 字节的 LFS pointer，不是完整权重。当前可用的完整主权重为 Q4_K_M。

### 4.2 本地准备验证

已用 Splash 的 Python 接口执行 `prepare_local_model()` 本地检查：

- 主 GGUF 通过；
- DFlash2 目录通过；
- 文本模式 `language_only=True` 通过；
- 不需要访问 Hub。

### 4.3 推荐启动命令：先做文本 smoke test

在 3.5 上执行：

```bash
mkdir -p ~/llm-deploy/{logs,run}

nohup /opt/homebrew/bin/splash serve \
  --model 'Qwen/Qwen3.8-27B' \
  --model-dir /Users/jiuwen/Models/splash/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --draft-model /Users/jiuwen/Models/splash/Qwen3.8-27B-DFlash2 \
  --language-only \
  --offline \
  --host 127.0.0.1 \
  --port 18080 \
  --max-context 8K \
  --max-memory 40G \
  --kv-format int8 \
  --state-format bf16 \
  --verify-rows all \
  --queue-size 1 \
  --request-timeout 30m \
  --idle-release 10m \
  > ~/llm-deploy/logs/splash-qwen38-smoke.log 2>&1 < /dev/null &

echo $! > ~/llm-deploy/run/splash-qwen38.pid
```

注意事项：

- `--model` 是必填项，即使使用本地文件也必须提供；这里只作为 API 显示 ID。
- 正式基线优先用文本模式，避免视觉 tower 影响内存和速度。
- `--max-memory 40G` 是保守起点，不是固定最优值；需要根据实际 RSS 和 peak memory 调整。
- `--max-context 8K` 仅用于 smoke test；正式测试再按矩阵提升。
- 若之后要做视觉评测，去掉 `--language-only`，并同时加入：

```bash
--mmproj /Users/jiuwen/Models/splash/Qwen3.8-27B-GGUF/mmproj-BF16.gguf
```

### 4.4 等待服务就绪

```bash
for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:18080/health >/dev/null 2>&1; then
    echo "Splash HTTP ready"
    break
  fi
  sleep 1
done

curl -fsS http://127.0.0.1:18080/ready
curl -fsS http://127.0.0.1:18080/status
```

就绪判断应以 `/ready` 成功、`/status` 中的模型和上下文信息正确为准，不能只看端口是否监听。

### 4.5 冒烟请求

```bash
curl -fsS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [{"role": "user", "content": "只回答：Splash Qwen3.8 加载成功"}],
    "max_tokens": 32,
    "temperature": 0
  }' | python3 -m json.tool
```

成功标准：

- HTTP 2xx；
- `choices[0].message.content` 有正常回答；
- `/status` 显示正确模型和上下文；
- 进程 RSS、峰值内存、swap 在可解释范围内；
- 日志无 `Engine failed`、EOF、OOM 或重复重启。

### 4.6 历史日志的风险提示

已有 `run.log` 中出现过：

```text
Engine failed · native protocol reached EOF
Engine restarted
```

同时也出现过成功加载与推理。该日志只能作为历史参考，不能证明当前配置稳定。正式测试前必须重新 smoke test，并保存新的启动命令和日志。

### 4.7 性能上限判断

- **速度上限**：在独占机器、Q4_K_M、DFlash2、INT8 KV 条件下可接近较好的本地速度表现。
- **质量上限**：不能宣称。Q4_K_M 不是最高质量量化，128K 也不是完整 262,144 原生窗口。
- **视觉能力**：资产有 mmproj，但正式视觉测试需要单独启动带视觉的实例并验证。

---

## 5. Qwen3.6-35B-A3B Splash 部署方案

### 5.1 当前资产

现有主模型：

```text
/Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit
```

- MLX 4bit affine；
- 约 19GiB；
- 4 个 safetensors 分片；
- `architectures: Qwen3_5MoeForConditionalGeneration`；
- `model_type: qwen3_5_moe`；
- 原生上下文 262,144；
- 40 层，30 层 linear attention、10 层 full attention；
- 256 routed experts，每 token 激活 8 个；
- 已通过 Splash 的 MLX 文本架构检查。

### 5.2 还缺什么

Splash 启动 Qwen3.6 时还需要匹配的 DFlash2 draft。当前机器只有 Qwen3.8 的 draft，不能跨模型复用。

上游 `incoai/Qwen3.6-35B-A3B-DFlash2` 已核实：

| 文件 | 大小 |
|---|---:|
| `config.json` | 1,291 bytes |
| `model.safetensors` | 1,052,513,456 bytes ≈ 0.98GiB |
| 仓库 revision | `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` |

因此本轮**不需要重新下载 19GB 主模型**，只需补约 1.05GB draft。

### 5.3 补 DFlash2

建议目录：

```text
/Users/jiuwen/Models/splash/Qwen3.6-35B-A3B-DFlash2
```

下载时应显式固定 revision，避免使用移动的默认分支：

```bash
mkdir -p /Users/jiuwen/Models/splash/Qwen3.6-35B-A3B-DFlash2

HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
  incoai/Qwen3.6-35B-A3B-DFlash2 \
  --revision 51ef7b6923ad6c14cb1bb41c37a9041446496ab6 \
  --include config.json \
  --include model.safetensors \
  --local-dir /Users/jiuwen/Models/splash/Qwen3.6-35B-A3B-DFlash2
```

下载后验证：

```bash
ls -lh /Users/jiuwen/Models/splash/Qwen3.6-35B-A3B-DFlash2
shasum -a 256 /Users/jiuwen/Models/splash/Qwen3.6-35B-A3B-DFlash2/model.safetensors
```

`config.json` 应包含 `DFlash2DraftModel`，并应匹配 Qwen3.6 的层几何；不要用 Qwen3.8 的 draft 代替。

### 5.4 推荐启动命令

```bash
nohup /opt/homebrew/bin/splash serve \
  --model 'mlx-community/Qwen3.6-35B-A3B-4bit' \
  --model-dir /Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit \
  --draft-model /Users/jiuwen/Models/splash/Qwen3.6-35B-A3B-DFlash2 \
  --language-only \
  --offline \
  --host 127.0.0.1 \
  --port 18081 \
  --max-context 8K \
  --max-memory 42G \
  --kv-format int8 \
  --state-format bf16 \
  --verify-rows all \
  --queue-size 1 \
  --request-timeout 30m \
  --idle-release 10m \
  > ~/llm-deploy/logs/splash-qwen36-smoke.log 2>&1 < /dev/null &

echo $! > ~/llm-deploy/run/splash-qwen36.pid
```

若希望与推荐 GGUF 基线完全一致，也可另下载并固定 `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` 的 GGUF 版本。两种版本必须分表，不能把 MLX 4bit 与 GGUF Q4_K_M 当成同一量化。

### 5.5 内存与上下文判断

主要内存来源：

- 约 19GiB 的 MLX 4bit 主权重；
- 约 1GiB DFlash2；
- KV Cache；
- Metal 临时缓冲；
- OS 与共享服务；
- 若启用视觉 tower，还会增加视觉权重。

在 64GB、单模型独占条件下：

- 8K、32K、64K、128K 是合理测试阶梯；
- 256K 需要单独探测，不应默认成功；
- 不应与当前 30GiB Ollama 35B 服务同时运行。

### 5.6 能否发挥最大性能

- **有条件可以接近速度上限**：必须独占机器、使用匹配 DFlash2、固定 KV 精度、避免 expert/权重走 SSD。
- **不能代表质量上限**：主权重是 4bit affine，不是 BF16/FP16。
- **结果标签要写清楚**：`Splash + MLX 4bit affine + DFlash2`，不要只写“Qwen3.6 性能”。

---

## 6. Gemma-4-26B-A4B MLX 部署方案

### 6.1 为什么必须走 MLX

Gemma 的配置为：

```text
architectures: Gemma4ForConditionalGeneration
model_type: gemma4
text_config.model_type: gemma4_text
max_position_embeddings: 262144
sliding_window: 1024
num_hidden_layers: 30
num_experts: 128
top_k_experts: 8
```

Splash 1.5.0 不支持该架构，且不能通过修改模型名或换目录绕过。因此按既定规划使用 MLX-VLM。

### 6.2 当前资产与运行环境

模型目录：

```text
/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit
```

约 14GiB，包含完整三片 safetensors、tokenizer、chat template、processor config 和 config。

Python 环境：

```text
/Users/jiuwen/mlx-bench-venv
mlx 0.32.4
mlx-vlm 0.8.0
transformers 5.24.0
huggingface-hub 2.4.0
Python 3.14.8
```

MLX-VLM 已包含 `gemma4` 与 `gemma4_text` 实现，具备加载条件。

### 6.3 加载验证

先做单次 CLI smoke test：

```bash
/Users/jiuwen/mlx-bench-venv/bin/mlx_vlm.generate \
  --model /Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit \
  --prompt "只回答：Gemma-4 加载成功" \
  --max-tokens 32 \
  --temperature 0.0 \
  --no-verbose
```

如果输出正常且进程退出，说明模型、tokenizer、Metal 和权重加载链路正常。

### 6.4 常驻 OpenAI 兼容服务

```bash
nohup /Users/jiuwen/mlx-bench-venv/bin/mlx_vlm.server \
  --host 127.0.0.1 \
  --port 18091 \
  --model /Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit \
  --max-tokens 2048 \
  --max-num-seqs 1 \
  --log-level INFO \
  > ~/llm-deploy/logs/gemma4-server.log 2>&1 < /dev/null &

echo $! > ~/llm-deploy/run/gemma4-server.pid
```

注意：

- `--max-num-seqs 1` 用于正式单请求测试，防止并发污染；
- `--max-tokens` 是单次生成上限，不是上下文窗口；
- 正式测试前等待 `/health` 返回 `model_ready=true`；
- 若使用 `--host 0.0.0.0`，应通过可信内网或 API key 限制访问。

等待就绪：

```bash
for i in $(seq 1 180); do
  health="$(curl -fsS http://127.0.0.1:18091/health 2>/dev/null || true)"
  printf '%s\n' "$health"
  if printf '%s' "$health" | python3 -c \
    'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("model_ready") is True else 1)' \
    2>/dev/null; then
    echo "Gemma-4 ready"
    break
  fi
  sleep 1
done
```

冒烟请求：

```bash
curl -fsS http://127.0.0.1:18091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit",
    "messages": [{"role": "user", "content": "只回答：Gemma-4 服务加载成功"}],
    "max_tokens": 32,
    "temperature": 0
  }' | python3 -m json.tool
```

### 6.5 已知冒烟基线

此前已完成的 MLX-VLM 冒烟验证结果：

| 指标 | Gemma-4-26B-A4B |
|---|---:|
| `/health` | `model_ready=true` |
| `/v1/models` | 通过 |
| Chat Completions | 通过 |
| 峰值内存 | 约 15.42GB |

该数据只证明加载链路与接口正常，不是正式性能基线。

### 6.6 能否发挥最大性能

- **速度上限**：可在 64GB 单模型独占条件下较充分释放 MLX/Metal 性能。
- **质量上限**：当前是 4bit affine 量化，不是 BF16 原版权重。
- **不能做的事**：不能把 MLX 结果填进 Splash 表，也不能用 Gemma 的 MLX tok/s 去和 Qwen 的 Splash tok/s 直接排名。

---

## 7. 跨引擎评测口径

### 7.1 必须分表

建议至少分三张表：

| 表 | 引擎 | 模型 |
|---|---|---|
| A | Splash 1.5.0 | Qwen3.8-27B |
| B | Splash 1.5.0 | Qwen3.6-35B-A3B |
| C | MLX-VLM 0.8.0 | Gemma-4-26B-A4B |

表中的每行至少应包含：

```text
engine
engine_version
model
model_revision
weight_format
quantization
context_limit
kv_format
draft_model
host
memory_gb
startup_command
```

### 7.2 可以横向比较的内容

- 同一引擎、同一机器、同一量化精度下的模型；
- 同一输入长度和输出长度下的 TTFT / Prefill / Decode；
- 同一上下文阶梯下的峰值内存和稳定性；
- 同一 warm-up 与重复次数后的统计分布。

### 7.3 不应横向比较的内容

- Splash 的 DFlash2 加速与 MLX-VLM 的普通 decode 直接比 tok/s；
- GGUF Q4_K_M 与 MLX 4bit affine 直接宣称“模型能力差异”；
- 不同 KV 精度、不同上下文、不同 reasoning effort 直接混表；
- 冷启动一次请求与 warm 后多次请求混在一起；
- 有并发与单请求结果混在一起；
- MLX/Ollama/Splash 三种引擎只写“某模型 tok/s”而不写引擎。

---

## 8. 上下文与性能测试矩阵

### 8.1 上下文阶梯

建议先测：

```text
1K
8K
32K
64K
128K
```

128K 稳定后，再单独探测：

```text
256K
```

解释：

- 1K 用于测短上下文 Decode 上限和框架开销；
- 8K 用于常规对话/评测集的代表长度；
- 32K 用于中等长文档；
- 64K 用于长上下文压力；
- 128K 是 64GB 机器上较现实的正式上限；
- 256K 是原生窗口边界探测，不是默认生产配置。

### 8.2 每个上下文点至少记录

| 指标 | 建议单位 | 说明 |
|---|---|---|
| cold load time | s | 进程启动到 `/ready` |
| TTFT | s / ms | 首个输出 token 延迟 |
| Prefill speed | tok/s | 输入 token 处理速度 |
| Decode speed | tok/s | 输出 token 生成速度 |
| prompt_tokens | tokens | 实际输入长度 |
| completion_tokens | tokens | 实际输出长度 |
| peak RSS | GiB | 进程峰值常驻内存 |
| wired memory | GiB | 锁定内存 |
| swap used | MiB/GiB | 是否发生换页 |
| SSD read/write | MiB/s 或总量 | 判断是否发生流式读取 |
| error/timeout | count | 稳定性 |
| finish_reason | string | 是否正常结束 |

### 8.3 固定参数

正式对比中必须固定：

```text
temperature = 0
top_p = 1
max_tokens = 固定值
stream = 固定
reasoning effort = 固定
KV format = 固定
context limit = 测试点
queue/concurrency = 1
warm-up = 至少 1 次
repeat = 3～5 次
```

### 8.4 建议测试顺序

1. Qwen3.8-27B，Splash，文本，1K smoke；
2. Qwen3.8-27B，1K/8K/32K/64K/128K；
3. Qwen3.8-27B，256K 边界探测（单独记录成功/失败）；
4. Qwen3.6-35B-A3B，补 DFlash2 后，重复同一矩阵；
5. Gemma-4-26B-A4B，MLX-VLM，同一输入/输出长度矩阵；
6. 汇总时按引擎分表，再做“同引擎内对比”和“跨引擎观察”，不直接合并排名。

---

## 9. 资源冲突与独占要求

### 9.1 当前冲突

| 冲突项 | 影响 |
|---|---|
| Ollama 35B 已驻留约 30GiB | 挤压 Splash/Qwen/Gemma 的 KV 与 Metal 缓冲 |
| MLX 9B 常驻约 5GiB | 额外占用统一内存和 GPU |
| NVIDIA PAIR 代理 | 可能继续触发 Ollama 请求 |
| SSD 与 swap 历史活动 | 影响长上下文和加载稳定性 |
| 其他同事共用 | 请求并发和系统负载不可控 |

### 9.2 建议的独占窗口

正式测量期间应满足：

- 与机器使用方约定独占时间；
- 停掉或隔离 NVIDIA PAIR 的模型负载；
- 停止 Ollama 已加载的 Qwen3.6；
- 停止 MLX-LM 的 Qwen3.5-9B；
- 一次只启动一个目标模型；
- 测完一个模型后完整退出进程，再启动下一个；
- 记录开始/结束时间、机器负载和异常事件。

### 9.3 不应直接做的事

- 不擅自 `kill` 同事的共享服务；
- 不在 Ollama 仍驻留 30GiB 时宣称 Splash 的正式 tok/s；
- 不把 MLX-LM 的 9B 结果与 Gemma/Qwen 混在同一张表；
- 不在磁盘写入、下载或索引构建期间做正式测速；
- 不将 `--max-memory` 设得过高以致系统 swap。

---

## 10. 风险与后续动作

### 10.1 风险清单

| 风险 | 级别 | 说明 | 处理 |
|---|---|---|---|
| 共享服务未释放 | 高 | 结果不可复现，内存/带宽受污染 | 协调独占窗口 |
| Qwen3.8 历史 Splash EOF | 中高 | 旧日志有引擎重启 | 重新 smoke test，保留新日志 |
| Qwen3.6 缺 DFlash2 | 中 | 无法走 Splash 的完整 draft 路径 | 下载固定 revision 的约 1.05GB draft |
| Qwen3.6 MLX 4bit 与 GGUF 混用 | 中 | 量化格式不同，不可混表 | 标注权重格式，必要时补 GGUF 对照 |
| Gemma 走 MLX | 低 | 架构不支持 Splash | 明确标注非 Splash |
| 256K OOM/swap | 中高 | 64GB 仍有上限 | 单独探测，不作为默认 |
| 长上下文 SSD 抖动 | 中 | 可能触发 KV/state 缓存写入 | 关闭或固定缓存策略并记录 |
| 视觉与文本混淆 | 中 | mmproj/visual tower 改变内存与速度 | 文本先 `--language-only`，视觉单独实验 |
| 量化质量上限被误读 | 中 | 4bit/Q4 不代表 BF16 能力 | 报告写清量化与上下文 |

### 10.2 后续动作清单

#### P0：正式测试前必须完成

- [ ] 与 3.5 使用方确认独占窗口；
- [ ] 停止 NVIDIA PAIR 模型负载、Ollama Qwen3.6 和 MLX-LM 9B；
- [ ] 确认 `18080`、`18081`、`18091` 无冲突；
- [ ] 记录当前 `vm_stat`、`memory_pressure`、swap、磁盘空间；
- [ ] 对 Qwen3.8 做 Splash 8K smoke test；
- [ ] 确认 `/ready`、`/status`、chat completion 和日志正常；
- [ ] 固定模型 revision、量化、KV、上下文和生成参数。

#### P1：Splash 模型

- [ ] Qwen3.8-27B 完成 1K/8K/32K/64K/128K 测试；
- [ ] Qwen3.8-27B 单独探测 256K；
- [ ] 下载并校验 Qwen3.6 DFlash2；
- [ ] Qwen3.6-35B-A3B 完成同一矩阵；
- [ ] 如需严格对照 GGUF，补 Qwen3.6 GGUF Q4_K_M 并单独标注；
- [ ] 记录 DFlash2 对 Decode 的影响；
- [ ] 记录 expert/权重是否发生 SSD 流式读取。

#### P1：Gemma MLX

- [ ] Gemma-4-26B-A4B MLX smoke test；
- [ ] 固定 `--max-num-seqs 1`；
- [ ] 完成 1K/8K/32K/64K/128K；
- [ ] 单独探测 256K；
- [ ] 记录 MLX peak memory、RSS、swap；
- [ ] 明确标注 `engine=MLX-VLM 0.8.0`。

#### P2：汇总

- [ ] 按引擎分表；
- [ ] 同引擎内比较模型；
- [ ] 跨引擎只做观察，不做直接排名；
- [ ] 报告冷加载、TTFT、Prefill、Decode、内存、上下文稳定性；
- [ ] 保存原始日志、命令、模型 revision 和测试日期；
- [ ] 对异常样本单独说明，不删除失败记录。

---

## 11. 最短执行路径

### Step 1：取得独占窗口并清理负载

只记录，不擅自停止：

```bash
ssh macmini-m5-jiuwen
ps -axo pid,rss,command | egrep 'ollama|mlx_lm|mlx_vlm|splash serve|llama-server' | egrep -v egrep
lsof -nP -iTCP -sTCP:LISTEN | egrep ':(8080|8090|11434|11435|18080|18081|18091)\b' || true
memory_pressure | tail -12
vm_stat | head -16
df -h /System/Volumes/Data
```

与使用方确认后，再停止对应服务。

### Step 2：Qwen3.8 Splash smoke

使用第 4 节的启动命令，先 `--max-context 8K`，确认 `/ready` 和一次 chat request。

### Step 3：Qwen3.8 正式矩阵

按 1K/8K/32K/64K/128K 执行，再单独探测 256K。

### Step 4：补 Qwen3.6 DFlash2

使用第 5 节固定 revision 下载命令，校验文件大小和哈希，再启动 Qwen3.6 Splash。

### Step 5：Gemma MLX

使用第 6 节 MLX-VLM 命令，固定单请求，执行同一矩阵。

### Step 6：分引擎汇总

将 Splash 与 MLX 分表记录，报告中始终保留引擎和量化标签。

---

## 12. 最终建议

1. **本轮首选在 3.5 上用 Splash 测 Qwen3.8-27B。** 资产完整，磁盘充足，最容易先建立干净的 Splash 基线。
2. **Qwen3.6 不必重下主模型。** 现有 19GB MLX 4bit 已通过 Splash 架构检查，只需补约 1.05GB DFlash2；若需要 GGUF 对照，再单独下载 GGUF。
3. **Gemma-4-26B-A4B 坚持 MLX-VLM。** Splash 1.5.0 明确不支持 `gemma4_text`，这不是配置问题。
4. **正式测试必须等独占窗口。** 当前约 30GiB Ollama + 约 5GiB MLX 服务会使任何 Splash/Gemma 性能结论失真。
5. **报告必须分引擎、分量化、分上下文。** 否则“哪个模型更快”的结论没有技术意义。

---

## 附录 A：文档与路径

| 内容 | 路径 |
|---|---|
| 本方案 | `/Users/augustus/Desktop/LLMBenchmark/docs/Jiuwen3.5_Splash与Gemma_MLX评测部署方案.md` |
| 3.13 Splash 四模型分析 | `/Users/augustus/Desktop/LLMBenchmark/docs/Splash四模型评测可行性分析.md` |
| MLX 部署操作手册 | `/Users/augustus/Desktop/LLMBenchmark/docs/MLX框架大模型部署完整操作手册.md` |
| MLX 加载与 OpenAI 服务 | `/Users/augustus/Desktop/LLMBenchmark/docs/MLX_模型加载与OpenAI服务操作手册.md` |

## 附录 B：本次核查证据摘要

```text
Host: Jiuwen-MacMini
Chip: Apple M5 Pro
CPU: 18 cores
GPU: 20 cores, Metal 4
Memory: 64 GB
macOS: 27.2
Data volume: 926 GiB, 454 GiB available
Splash: 1.5.0
MLX runtime: mlx 0.32.4, mlx-vlm 0.8.0, transformers 5.24.0
Qwen3.8 GGUF: 16,464,440,224 bytes
Qwen3.8 mmproj: 931,146,432 bytes
Qwen3.8 DFlash2: 3,848,817,896 bytes
Qwen3.6 MLX model: about 19 GiB, 4 shards
Qwen3.6 DFlash2 upstream model.safetensors: 1,052,513,456 bytes
Gemma-4 MLX model: about 14 GiB, 3 shards
```

## 附录 C：术语

| 术语 | 含义 |
|---|---|
| TTFT | Time To First Token，从请求发出到首个输出 token 的时间 |
| Prefill | 处理输入 prompt 的阶段，常以 tok/s 表示 |
| Decode | 逐 token 生成输出的阶段，常以 tok/s 表示 |
| RSS | Resident Set Size，进程常驻内存 |
| wired memory | 不可换出的锁定内存 |
| KV Cache | 注意力键值缓存，长上下文时显著增长 |
| DFlash2 | Splash 使用的投机解码 draft |
| MLX | Apple Silicon 上的机器学习数组/推理框架 |
| MLX-VLM | 基于 MLX 的视觉语言模型推理框架 |
| LFS pointer | Git LFS 指针文件，不是实际模型权重 |
