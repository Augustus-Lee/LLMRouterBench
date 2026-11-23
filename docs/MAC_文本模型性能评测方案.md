# Mac mini M4/M5 文本模型性能摸底评测方案（第一阶段：文本模型）

> 项目仓库：`/Users/augustus/Desktop/LLMBenchmark`
> 评测目标：Qwen3.8-27B、Qwen3.8-Flash-125B-A6B、Qwen3.6-35B-A3B、Gemma-4-26B-A4B
> 评测维度：Prefill 速度、Decode 速度、内存占用、最大可用上下文、机器性能影响分析
> 执行方式：**远程连接两台 Mac mini M5 Pro 64GB 与一台 Mac mini M4 Pro 64GB**
> 日期：2026-11-23（v3，校正实机配置并补充 Qwen3.6-35B-A3B 的 Ollama / MLX 选型）

---

## 一、重要前置：硬件可行性分析

### 1.1 目标硬件规格

| 项目 | Mac mini M4 Pro（192.168.3.7） | Mac mini M5 Pro（192.168.3.5 / .13） |
|---|---|---|
| 架构 | Apple Silicon (ARM64) | Apple Silicon (ARM64) |
| 型号标识 | Mac16,11 | Mac17,16 |
| CPU | 14 核（10 性能 + 4 能效） | 18 核（6 性能 + 12 能效） |
| GPU | 20 核 | 20 核 |
| 统一内存 | **64 GB** | **64 GB** |
| 内存带宽 | 约 273 GB/s | 待用带宽专项测试实测确认 |
| 神经引擎 | 16 核 | 16 核 |
| 加速栈 | MLX / Metal / llama.cpp Metal | MLX / Metal / llama.cpp Metal |
| 系统盘可用空间 | 约 3.0 TiB | 3.5 约 489 GiB；3.13 约 63 GiB |

### 1.2 可行性结论（v3 更新）

| 模型 | 参数量 | FP16 内存 | INT4 内存 | M4 Pro (64GB) | M5 Pro (64GB) |
|---|---:|---:|---:|---|---|
| Qwen3.8-27B | 27B | ~54 GB | **~15–17 GB** | ✅ 舒适 | ✅ 舒适 |
| Qwen3.8-Flash-125B-A6B | 125B（激活 6B） | ~250 GB | **~65–80 GB** | ❌ Q4 不适合作为稳定基线；仅可做 Q2/Q3 短上下文实验 | ❌ Q4 同样不适合作为稳定基线；仅可做低比特实验 |
| Qwen3.6-35B-A3B | 35B（激活 3B） | ~70 GB | **~20 GB** | ✅ 舒适 | ✅ 舒适 |
| Gemma-4-26B-A4B | 26B（激活 4B） | ~52 GB | **~15 GB** | ✅ 舒适 | ✅ 舒适 |

**结论：**

- **前三个 20B～35B 级模型在 64GB 机器上均适合做 4bit 评测；125B-A6B 的 Q4 不应作为正式基线。**
- 不能只按芯片代际推断性能；M4 Pro 与 M5 Pro 的 GPU 核数相同，Decode、Prefill、上下文上限都必须实测。
- Prefill 和 Decode 均主要运行在 Metal GPU 上，ANE 不是 MLX-LM 的主要执行单元。
- **推理引擎按目标选择：**模型能力与路由评测优先 Ollama；Mac 原生性能专项优先 MLX；正式引擎对比两者都测。

### 1.3 连接方式

从本机通过已配置的 SSH key 免密连接：

```bash
ssh macmini-m4-admin    # 192.168.3.7  / M4 Pro / 64GB
ssh macmini-m5-jiuwen   # 192.168.3.5  / M5 Pro / 64GB
ssh macmini             # 192.168.3.13 / M5 Pro / 64GB
```

如需图形界面，可使用 Screen Sharing，但正式性能测试应通过 SSH 执行，避免 GUI 活动干扰。

**建议将评测脚本托管在 git 仓库中，M4/M5 上 `git pull` 即可获得最新评测代码。**

---

## 二、评测维度定义

本次评测要回答 5 个核心问题，每个问题对应一个指标和测量方法。

### 2.1 Prefill 速度（TTFT / Prefill tokens/sec）

**定义：** 输入 N 个 prompt tokens 后，模型吐出第一个 token 的时间（TTFT），或单位时间内能处理的输入 token 数。

**测法：**

```text
TTFT = first_token_time - request_start_time
Prefill Speed = input_tokens / TTFT   (单位: tokens/sec)
```

**为什么重要：** 决定了"用户发送 2000 token 的问题后，等多久能看到第一个字"。

### 2.2 Decode 速度（Tokens Per Second）

**定义：** 模型逐 token 生成时的吞吐速度，即"每秒能写出多少 token"。

**测法：**

```text
Decode Speed = output_tokens / (last_token_time - first_token_time)
```

**为什么重要：** 决定长输出的“打字机”体验。Apple Silicon 上主要反映 GPU 算力和内存带宽，不应默认归因于 ANE。

### 2.3 内存占用

**定义：** 模型推理期间进程的常驻内存峰值（RSS / Memory Footprint）。

**测法：**

Apple Silicon 上建议同时采集“推理框架自报峰值”和“系统内存压力”：

```bash
# MLX CLI 会直接报告 Peak memory
mlx_lm.generate --model <MODEL> --prompt "test" --max-tokens 16

# 进程 RSS 采样
python3 -m pip install psutil
python3 - <<'RSS'
import psutil, time
proc = psutil.Process(<PID>)
peak = 0
while proc.is_running():
    peak = max(peak, proc.memory_info().rss)
    time.sleep(0.5)
print(f"Peak RSS: {peak / 1024**3:.2f} GiB")
RSS

# 系统级内存压力与 swap
memory_pressure
vm_stat
sysctl vm.swapusage
```

不要直接照搬 Linux 的 `/usr/bin/time -v`；macOS 自带 `/usr/bin/time` 不支持该参数。

**为什么重要：** 决定了"这个模型能不能塞进 64GB 统一内存"，以及"塞进去之后还剩多少内存跑 KV cache"。

### 2.4 最大支持上下文

**定义：** 在当前硬件下，模型能实际跑起来的最大 `n_ctx`（KV cache 长度）。

**测法：** 从小到大逐步提高 `n_ctx`（1024 → 2048 → 4096 → 8192 → 16384 → 32768 → 65536 → 131072 ...），观察何时 OOM 或崩溃。

### 2.5 机器性能影响分析

**定义：** 分析 CPU/GPU/神经引擎频率、内存带宽、温控降频、统一内存分配对推理速度的影响。

**测法：** 见第七节。

---

## 三、推理引擎选型（v3：按评测目标选择）

### 3.1 引擎对比

| 候选 | Apple Silicon 支持 | 易用性 | 优势 | 局限 | 适用场景 |
|---|---|---|---|---|---|
| **MLX-LM** | ✅ 原生 | ⭐⭐⭐⭐ | 直接使用 Apple Silicon GPU 和统一内存；可输出 Prompt/Generation tokens/s 与 Peak memory；支持 KV Cache 量化 | 只能加载 MLX 权重；MoE 对版本和模型结构较敏感；需单独下载约 20GB 权重 | Mac 原生推理性能、Prefill/Decode、峰值内存专项 |
| **llama.cpp (Metal)** | ✅ GPU 加速 | ⭐⭐⭐⭐ | GGUF 生态成熟，MoE 支持较广，参数可精细控制 | 需要自行编译或管理二进制与服务参数 | 底层性能复核、GGUF 量化对照 |
| **Ollama** | ✅ GPU 加速 | ⭐⭐⭐⭐⭐ | 三台机器已安装同一模型；API 统一；支持 tools/thinking/vision；复现成本最低 | 默认参数不透明；不同机器端口不一致；默认上下文不等于模型上限 | 模型能力、工具调用、跨机器统一基线 |
| **LM Studio** | ✅（底层 llama.cpp/MLX） | ⭐⭐⭐⭐⭐ | GUI 调试方便，支持 GGUF/MLX | 自动化与精细计时不如命令行稳定 | 交互式验证、非正式测试 |
| **vLLM** | ⚠️ Metal 后端不成熟 | ⭐⭐ | 服务化能力强 | Apple Silicon 不适合作为本轮基线 | 本轮不推荐 |

### 3.2 最终选型：按评测目标分工

**结论先说：Qwen3.6-35B-A3B 当前应“先用 Ollama 跑通能力和跨机基线，再用 MLX 做性能专项”，而不是二选一。**

原因有三点：

1. **Ollama 已具备可直接评测的条件。** 三台机器已安装同一个 `qwen3.6:35b-a3b-q4_K_M`（版本 0.33.3、同一 digest），无需额外下载，适合先接入 LLMRouterBench、验证模型能力和建立 M4/M5 统一性能基线。
2. **MLX 更适合做 Mac 原生性能研究，但当前还需准备权重。** 3.5 与 3.13 已具备支持 `qwen3_5_moe` 的 MLX-LM，但 Qwen3.6 的 MLX 4bit 权重尚未下载（约 20.4GB）；因此 MLX 是第二阶段专项，不应阻塞第一轮评测。
3. **不能把 Ollama 与 MLX 当成同一模型精度直接横向排名。** Ollama 使用 GGUF Q4_K_M，MLX 使用 affine 4bit（group size 64，部分门控层 8bit）；两者适合比较速度、内存和部署成本，但质量结果必须分别标注。

按下表选择具体入口：

| 评测目标 | 推荐引擎 | 原因 |
|---|---|---|
| 尽快评测 Qwen3.6-35B-A3B 模型能力 | **Ollama 为主** | 三台机器已有同一 Q4_K_M 模型，API 统一，接入成本最低 |
| 接入 LLMRouterBench 数据采集与路由回放 | **Ollama 为主** | OpenAI/Ollama API 简单，便于统一 prompt、采样和失败处理 |
| Mac mini 的 Prefill、Decode、峰值内存专项 | **MLX 为主** | 原生 MLX 路径，CLI 直接输出 Prompt/Generation tokens/s 与 Peak memory |
| M4 Pro 与 M5 Pro 跨机器对比 | **Ollama 做统一基线** | 三台机器可使用相同 GGUF 权重、量化与请求参数 |
| 引擎效率与量化研究 | **MLX + Ollama 都测** | 可区分引擎差异与量化差异，避免单引擎偏差 |
| 工具调用、Thinking、视觉能力 | **Ollama 优先** | 当前模型已声明 tools、thinking、vision 能力 |

**执行顺序：**

```text
阶段 1：Ollama 统一基线
  → 三台机器跑同一 qwen3.6:35b-a3b-q4_K_M
  → 固定 num_ctx、num_predict、temperature、seed
  → 建立能力与性能基线

阶段 2：MLX 性能专项
  → 在 192.168.3.5 下载并加载 mlx-community/Qwen3.6-35B-A3B-4bit
  → 测 Prefill / Decode / Peak memory / 上下文阶梯
  → Ollama 同配置交叉验证

阶段 3：M4 / M5 对照
  → M4 Pro 使用 Ollama 统一基线
  → M5 Pro 使用 Ollama + MLX
  → 分开报告“硬件差异”和“引擎差异”
```

### 3.3 Qwen3.6-35B-A3B 的当前实机状态

#### Ollama

三台机器均已有：

```text
qwen3.6:35b-a3b-q4_K_M
```

关键元数据：

```text
格式：GGUF
量化：Q4_K_M
参数：约 36.0B
架构：qwen35moe
专家数：256
每 token 使用专家数：8
逻辑模型大小：约 23.9GB
上下文元数据：262144
能力：completion、vision、tools、thinking
Ollama 版本：0.33.3
```

实际 API 端口：

| 机器 | 实际 Ollama | 注意事项 |
|---|---|---|
| 192.168.3.5 M5 Pro 64GB | `127.0.0.1:11435` | 11434 是 NVIDIA PAIR 代理，不是底层 Ollama |
| 192.168.3.13 M5 Pro 64GB | `127.0.0.1:11435` | 同上 |
| 192.168.3.7 M4 Pro 64GB | `127.0.0.1:11434` | Ollama.app 直接监听 11434 |

因此，在 3.5 和 3.13 上做正式性能测试时，应直连 `11435`，避免 PAIR 代理增加额外变量。

#### MLX-LM

已确认：

```text
192.168.3.5   mlx-lm 0.32.0 / mlx 0.32.3
192.168.3.13  mlx-lm 0.32.2 / mlx 0.32.4
```

两者都包含 `qwen3_5_moe` 模型实现，说明代码层面支持该 MoE 架构。

公开 MLX 权重：

```text
mlx-community/Qwen3.6-35B-A3B-4bit
```

已核查：

```text
model_type：qwen3_5_moe
architecture：Qwen3_5MoeForConditionalGeneration
量化：4bit，group size 64
部分门控层：8bit
safetensors：4 个文件
权重总量：约 20.4GB
访问权限：公开，非 gated
revision：38740b847e4cb78f352aba30aa41c76e08e6eb46
```

**但当前三台机器都没有安装这个 Qwen3.6 MLX 权重。**

现有目录：

```text
~/Models/Qwen3.5-35B-A3B-4bit
```

这是 **Qwen3.5**，不是 Qwen3.6，不能作为 Qwen3.6 的 MLX 测试对象。

### 3.4 关键注意事项

1. **GGUF Q4_K_M 与 MLX 4bit 不是同一量化。**
   - Ollama 当前为 GGUF Q4_K_M；
   - MLX 为 affine 4bit，group size 64，部分门控层 8bit；
   - 可以比较速度、内存和部署成本，但不能把质量结果当成完全相同的模型精度。

2. **模型元数据的 262144 上下文不代表运行时已启用 262K。**
   - Ollama 必须显式设置 `options.num_ctx`；
   - MLX 必须记录实际 KV Cache 长度和内存峰值；
   - 第一轮建议只测 8K、32K、64K。

3. **Qwen3.6 MLX 权重属于多模态 ConditionalGeneration 结构。**
   - 当前 MLX-LM 能识别 `qwen3_5_moe` 并在加载时过滤视觉权重；
   - 正式下载后仍必须做一次最小加载与文本生成验证；
   - 如果加载失败，应升级 MLX-LM 或改用支持该模型的 mlx-vlm 路径，不能直接假定可用。

4. **MLX 服务端口不要使用 8080。**
   - 3.5 和 3.13 的 8080 已被 NVIDIA PAIR 占用；
   - 建议 MLX 使用 `127.0.0.1:8081`。

### 3.5 模型格式与下载

| 模型 | MLX 格式（性能专项） | GGUF / Ollama 格式（统一基线） |
|---|---|---|
| Qwen3.8-27B | mlx-community/Qwen3.8-27B-4bit | Qwen3.8-27B-Q4_K_M.gguf |
| Qwen3.8-Flash-125B-A6B | 仅低比特实验，不作为 Q4 基线 | 仅低比特实验，不作为 Q4 基线 |
| Qwen3.6-35B-A3B | mlx-community/Qwen3.6-35B-A3B-4bit | 三台已装 Ollama `qwen3.6:35b-a3b-q4_K_M` |
| Gemma-4-26B-A4B | mlx-community/gemma-4-26b-a4b-4bit | Q4_K_M.gguf |

> MLX 权重会额外占用约 20GB 磁盘。未经确认不要批量下载；先以 Ollama 基线跑通评测流水线。

---

## 四、评测流程总览

```text
[Step 0] 环境与模型状态核查
  → 确认三台机器的 Ollama 版本、模型 digest 与实际端口
  → 确认 MLX-LM 版本及 qwen3_5_moe 支持
  → 检查磁盘、可用内存和已有后台进程

[Step 1] Ollama 统一基线（优先执行）
  → 三台机器统一使用 qwen3.6:35b-a3b-q4_K_M
  → 显式固定 num_ctx / num_predict / temperature / seed
  → 预热 1 次，正式重复 3 次

[Step 2] MLX 性能专项
  → 在 192.168.3.5 下载 Qwen3.6 MLX 4bit 权重
  → 先做最小加载与文本生成验证
  → 测 Prefill / Decode / 峰值内存 / 上下文阶梯

[Step 3] 引擎交叉验证
  → 同一 prompt、同一上下文、同一输出长度
  → 对比 Ollama Q4_K_M 与 MLX 4bit
  → 结果按引擎和量化分开报告

[Step 4] M4 Pro vs M5 Pro 对比
  → 优先使用相同的 Ollama GGUF 基线
  → 再在可用机器上补充 MLX 对照

[Step 5] 汇总输出
  → JSON 结果 → Markdown 报告 → 图表
```

---

## 五、详细评测脚本设计

### 5.1 环境准备

现有环境已经具备 Homebrew 版 MLX-LM。正式评测前先核查，而不是重复安装：

```bash
ssh macmini-m5-jiuwen

command -v mlx_lm.generate
mlx_lm.generate --help | head

/opt/homebrew/opt/mlx-lm/libexec/bin/python - <<'PY'
import importlib.metadata as md
import pkgutil
import mlx_lm.models as models

print("mlx-lm:", md.version("mlx-lm"))
print("mlx:", md.version("mlx"))
print("qwen modules:", [m.name for m in pkgutil.iter_modules(models.__path__) if "qwen" in m.name])
PY
```

如果使用项目内的独立 Python 环境，再执行：

```bash
cd ~/LLMRouterBench
uv venv .venv-mlbench --python 3.11
source .venv-mlbench/bin/activate
uv pip install mlx-lm huggingface_hub psutil openai httpx matplotlib pandas
```

注意：系统默认 `python3` 当前不能直接 `import mlx_lm`，但 Homebrew 的 `mlx_lm.*` 命令可用。评测脚本必须固定解释器与 MLX-LM 版本，不能混用两个环境。

### 5.2 模型准备

#### 第一步：直接复用 Ollama 模型（推荐先做）

三台机器已安装：

```text
qwen3.6:35b-a3b-q4_K_M
```

无需重新下载。先验证：

```bash
# 3.5 / 3.13
curl -fsS http://127.0.0.1:11435/api/tags | python3 -m json.tool

# 3.7
curl -fsS http://127.0.0.1:11434/api/tags | python3 -m json.tool
```

#### 第二步：按需下载 Qwen3.6 MLX 权重

目标仓库：

```text
mlx-community/Qwen3.6-35B-A3B-4bit
revision: 38740b847e4cb78f352aba30aa41c76e08e6eb46
实际文件总量：约 20.4GB
```

推荐只在 `192.168.3.5` 下载，先确认磁盘空间：

```bash
ssh macmini-m5-jiuwen
df -h ~
mkdir -p ~/Models
```

下载命令如下。**该命令会下载约 20GB，执行前需要确认：**

```bash
/opt/homebrew/opt/mlx-lm/libexec/bin/hf download \
  mlx-community/Qwen3.6-35B-A3B-4bit \
  --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --local-dir ~/Models/Qwen3.6-35B-A3B-4bit
```

> 不要下载 Qwen3.5-35B-A3B 后改目录名冒充 Qwen3.6。现有 `~/Models/Qwen3.5-35B-A3B-4bit` 属于 Qwen3.5。

### 5.3 Ollama 基线请求

在 3.5 / 3.13 上通过 SSH 隧道访问：

```bash
ssh -N -L 11435:127.0.0.1:11435 macmini-m5-jiuwen
```

另开终端发送请求：

```bash
curl -sS http://127.0.0.1:11435/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6:35b-a3b-q4_K_M",
    "messages": [
      {"role": "user", "content": "请用三句话解释 MoE 模型。"}
    ],
    "stream": false,
    "think": false,
    "options": {
      "num_ctx": 32768,
      "num_predict": 128,
      "temperature": 0,
      "seed": 42
    }
  }'
```

必须记录：

- `num_ctx` 实际值；
- `num_predict`；
- 是否启用 thinking；
- Ollama 版本；
- 模型 digest；
- 是否发生 swap；
- 同一次请求的 `prompt_eval_count` 与 `eval_count`。

### 5.4 启动 MLX 推理服务

3.5 / 3.13 的 8080 已被 NVIDIA PAIR 占用，MLX 服务统一使用 8081：

```bash
mlx_lm.server \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --host 127.0.0.1 \
  --port 8081 \
  --temp 0.0 \
  --max-tokens 512
```

最小加载验证：

```bash
mlx_lm.generate \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --prompt "请用三句话解释 MoE 模型。" \
  --max-tokens 128 \
  --temp 0 \
  --seed 42
```

预期 CLI 会输出：

```text
Prompt: N tokens, X tokens-per-sec
Generation: N tokens, Y tokens-per-sec
Peak memory: Z GB
```

这三个值分别作为 Prefill 速度、Decode 速度和峰值内存的第一手数据。

**MLX 验证门槛：**

1. 权重能完整加载；
2. 能完成一次文本生成；
3. 输出中没有视觉/音频占位 token 异常；
4. 峰值内存未导致系统 swap；
5. 上下文阶梯测试能稳定完成。

任何一项失败，都不能把该结果作为有效 MLX 基线。

### 5.5 Prefill / Decode 测量

#### MLX CLI

`mlx_lm.generate` 自带计时汇总，适合作为最小复核：

```bash
mlx_lm.generate \
  --model ~/Models/Qwen3.6-35B-A3B-4bit \
  --prompt - \
  --max-tokens 32 \
  --temp 0 \
  --seed 42 < prompts/prefill_2k.txt
```

#### Ollama API

Ollama `/api/chat` 返回中的 `prompt_eval_count`、`eval_count` 可用于计算：

```text
Prefill tok/s ≈ prompt_eval_count / prompt_eval_duration
Decode tok/s  ≈ eval_count / eval_duration
```

其中 duration 字段是纳秒，需要除以 `1e9`。为得到 TTFT 和流式 Decode 速度，还应另做一遍 `stream: true` 测试，并以客户端时间戳记录首 token 和末 token。

注意：不能假定 `mlx_lm.server` 与 llama.cpp 返回相同的 `timings` 字段。脚本应分别适配引擎，不要把缺失字段当成 0 分。

### 5.6 内存占用测量

macOS 自带的 `/usr/bin/time` 不支持 GNU `-v`，不要直接使用 Linux 命令。

推荐同时记录三类指标：

1. **MLX 自报 Peak memory**

```text
mlx_lm.generate 输出中的 Peak memory
```

2. **进程 RSS 采样**

```bash
python3 - <<'RSS'
import psutil, time
proc = psutil.Process(<PID>)
peak = 0
while proc.is_running():
    peak = max(peak, proc.memory_info().rss)
    time.sleep(0.5)
print(f"Peak RSS: {peak / 1024**3:.2f} GiB")
RSS
```

3. **系统内存压力与 swap**

```bash
memory_pressure
vm_stat
sysctl vm.swapusage
```

报告中要区分：

- 权重常驻内存；
- 推理进程峰值 RSS；
- KV Cache 增量；
- 系统 swap 使用量；
- 是否触发内存压力警告。

### 5.7 最大上下文测量

不要直接从 262K 开始。建议阶梯：

```text
8K → 16K → 32K → 64K → 128K
```

每一档记录：

- 实际输入 token 数；
- 是否成功生成；
- TTFT；
- Decode tok/s；
- 峰值内存；
- swap 增量；
- 是否 OOM 或进程崩溃。

Ollama 通过 `options.num_ctx` 设置上下文；MLX 通过 `--max-kv-size`、prompt 长度和进程峰值内存共同判断实际可用上限。模型配置中的 `max_position_embeddings` 只是架构上限，不等于本机可用上限。

### 5.8 完整测量脚本设计

建议把 Ollama 与 MLX 分成两个适配器，共享 prompt 集和结果 schema：

```text
tools/
  bench_common.py       # prompt 生成、统计、JSON schema
  bench_ollama.py       # Ollama /api/chat 适配器
  bench_mlx.py          # mlx_lm.generate 或 mlx_lm.server 适配器
  summarize.py          # 汇总 Markdown 与图表
```

统一记录字段至少包括：

```json
{
  "engine": "ollama",
  "engine_version": "0.33.3",
  "model": "qwen3.6:35b-a3b-q4_K_M",
  "model_revision_or_digest": "07d352...",
  "quantization": "Q4_K_M",
  "context_tokens": 32768,
  "max_output_tokens": 128,
  "temperature": 0,
  "seed": 42,
  "prefill_tok_per_s": null,
  "decode_tok_per_s": null,
  "ttft_s": null,
  "peak_memory_gib": null,
  "swap_used_gib": null,
  "success": true
}
```

同一份报告里不要把 `engine`、`quantization` 合并成一个模糊的“4bit”字段。

### 5.9 远程 / API 模型测量（备用方案）

只有当 64GB 机器也无法稳定加载 125B 模型时，才走云端 API 路线。

API 测出的 TTFT 包含网络往返，Decode 速度受服务商集群调度影响，不能用于推断本机硬件性能。报告中必须单列“云端 API 基线”，不能与本地 MLX/Ollama 结果混合排名。

---

## 六、M4 Pro vs M5 Pro 对比方案

本轮实际机器均为 **64GB 统一内存**，因此不能沿用“M5 内存更大所以能跑更长上下文”的旧假设。两者最大上下文限制主要由运行时、KV Cache 实现、可用内存余量和引擎配置共同决定。

### 6.1 Qwen3.6-35B-A3B 测试矩阵

| 测试项 | M4 Pro 64GB（192.168.3.7） | M5 Pro 64GB（192.168.3.5） | M5 Pro 64GB（192.168.3.13） |
|---|---|---|---|
| Ollama Q4_K_M | ✅ 已安装 | ✅ 已安装 | ✅ 已安装 |
| Ollama `num_ctx=8192` | 计划测试 | 计划测试 | 计划测试 |
| Ollama `num_ctx=32768` | 计划测试 | 计划测试 | 计划测试 |
| Ollama `num_ctx=65536` | 计划测试 | 计划测试 | 计划测试 |
| MLX 4bit | ⚠️ 需安装 MLX-LM 与权重 | ✅ 运行时已具备，待下载权重 | ⚠️ 先释放 Splash 与磁盘 |
| 主用途 | M4 硬件基线 | MLX 主测机 | 第二台 M5 复核机 |

### 6.2 对比原则

- **硬件差异**只比较相同引擎、相同模型、相同量化、相同上下文与输出长度。
- **引擎差异**只比较同一台机器上的 Ollama 与 MLX。
- 不同时改变“芯片 + 引擎 + 量化”，否则无法归因。
- 3.13 当前有 Splash 常驻，正式测试前必须记录或释放资源；否则不适合与 3.5 直接比较。
- 3.7 当前未发现 `mlx_lm.generate`，若要做 MLX 对照，需要单独安装运行时并下载权重。

### 6.3 需要验证的假设

不能预设“M5 一定快于 M4”，应逐项验证：

1. 在 **Ollama Q4_K_M** 相同条件下，M5 Pro 的 Decode tok/s 是否高于 M4 Pro；
2. 在相同 prompt 长度下，M5 Pro 的 Prefill tok/s 是否更高；
3. 两者在 8K、32K、64K 下的 KV Cache 内存增量是否不同；
4. 连续运行时的热降频是否导致速度衰减；
5. MLX 与 Ollama 的排名是否一致，还是引擎实现改变了硬件差异。

---

## 七、性能归因分析

### 7.1 关键硬件与运行时因素

| 因素 | 对 Prefill 的影响 | 对 Decode 的影响 | 本轮如何测量 |
|---|---|---|---|
| 内存带宽 | 高 | **很高，通常是 Decode 主瓶颈** | 同模型、同引擎、同上下文对比 tok/s |
| GPU 核数与架构 | **很高** | 中高 | M4 Pro / M5 Pro 同模型对比 |
| 统一内存容量 | 中 | 高，决定 KV Cache 上限 | 上下文阶梯和 OOM 点 |
| KV Cache 精度与实现 | 中 | 高 | 记录 FP16/INT8 等设置，不混测 |
| MoE 专家路由效率 | 高 | 高 | 观察不同引擎、不同 batch 下的差异 |
| 温控降频 | 中 | 中高 | 连续 Decode 多次，比较首轮与末轮 |
| 磁盘 I/O | 仅冷启动和换页时明显 | 发生 swap 时很高 | 记录冷/热启动与 swap 增量 |
| ANE | 低 | 低 | MLX-LM 主要使用 GPU，不把 ANE 当作主因 |

### 7.2 归因测试方法

```bash
# 查看当前硬件和系统
sysctl -n hw.model hw.memsize
system_profiler SPDisplaysDataType | egrep 'Chipset Model|Total Number of Cores|Metal'

# 查看统一内存与 swap
vm_stat
sysctl vm.swapusage
memory_pressure

# 观察推理进程的内存（把 PID 换成实际进程号）
ps -o pid,rss,vsz,command -p <PID>

# 观察 GPU/CPU 功耗与热状态，需要 sudo
sudo powermetrics --samplers cpu_power,gpu_power,thermal -i 1000
```

> `powermetrics` 需要管理员权限。没有权限时，不能编造 GPU 频率或温度，只能报告“未采集”。

### 7.3 结论模板

正式报告应回答：

1. **Decode 瓶颈**：Qwen3.6-35B-A3B 在 M4 Pro 与 M5 Pro 上的 Decode tok/s 差异是多少；这是带宽、内核实现还是热降频导致。
2. **Prefill 瓶颈**：不同输入长度下 TTFT 与 Prefill tok/s 如何变化；M5 的 GPU 架构是否带来稳定提升。
3. **内存瓶颈**：权重、KV Cache、运行时分别在峰值内存中占多少；是否出现 swap。
4. **上下文上限**：8K、32K、64K 哪些稳定通过；失败发生在加载、Prefill 还是 Decode。
5. **引擎差异**：同一台机器上 Ollama Q4_K_M 与 MLX 4bit 的 Prefill/Decode/内存差异；不能把量化差异误写成引擎差异。
6. **稳定性**：重复测试的均值、标准差和最差一次，而不只报告最好成绩。

---

## 八、输出格式

### 8.1 单次运行 JSON 结果

```json
{
  "engine": "ollama",
  "engine_version": "0.33.3",
  "model": "qwen3.6:35b-a3b-q4_K_M",
  "model_revision_or_digest": "07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522",
  "quantization": "Q4_K_M",
  "hardware": {
    "chip": "Apple M5 Pro",
    "model_identifier": "Mac17,16",
    "memory_gib": 64
  },
  "context_tokens": 32768,
  "max_output_tokens": 128,
  "temperature": 0,
  "seed": 42,
  "prefill": {
    "prompt_tokens": null,
    "ttft_s": null,
    "tok_per_s": null
  },
  "decode": {
    "output_tokens": null,
    "duration_s": null,
    "tok_per_s": null
  },
  "memory": {
    "peak_process_gib": null,
    "swap_delta_gib": null,
    "memory_pressure": null
  },
  "success": true,
  "notes": ""
}
```

所有 `null` 必须在正式运行后替换为实测值；缺失字段不能补 0，也不能估算。

### 8.2 汇总 Markdown 模板

```markdown
| 机器 | 引擎 | 模型/量化 | 上下文 | Prefill tok/s | Decode tok/s | 峰值内存 | 结果 |
|---|---|---|---:|---:|---:|---:|---|
| M4 Pro 64GB | Ollama | Qwen3.6-35B-A3B / Q4_K_M | 8192 | TBD | TBD | TBD | TBD |
| M4 Pro 64GB | Ollama | Qwen3.6-35B-A3B / Q4_K_M | 32768 | TBD | TBD | TBD | TBD |
| M5 Pro 64GB | Ollama | Qwen3.6-35B-A3B / Q4_K_M | 8192 | TBD | TBD | TBD | TBD |
| M5 Pro 64GB | Ollama | Qwen3.6-35B-A3B / Q4_K_M | 32768 | TBD | TBD | TBD | TBD |
| M5 Pro 64GB | MLX-LM | Qwen3.6-35B-A3B / MLX 4bit | 8192 | TBD | TBD | TBD | TBD |
```

### 8.3 对比汇总表模板

| 机器 | 引擎 | 量化 | 上下文 | Prefill tok/s | Decode tok/s | 峰值内存 GiB | swap GiB | 成功 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| M4 Pro 64GB | Ollama | Q4_K_M | 8192 | TBD | TBD | TBD | TBD | TBD |
| M5 Pro 64GB | Ollama | Q4_K_M | 8192 | TBD | TBD | TBD | TBD | TBD |
| M5 Pro 64GB | MLX-LM | MLX 4bit | 8192 | TBD | TBD | TBD | TBD | TBD |

对比时必须同时显示 `引擎` 和 `量化`。不能只写“Qwen3.6 35B 4bit”后直接比较。

### 8.4 图表输出

- **图表 1**：Prefill TTFT 与 Prefill tok/s 随输入长度变化；
- **图表 2**：Decode tok/s 随输出长度变化；
- **图表 3**：峰值内存与 KV Cache 增量随上下文变化，并标记 OOM；
- **图表 4**：Ollama 与 MLX 在同机、同上下文下的速度与内存对比；
- **图表 5**：M4 Pro 与 M5 Pro 在相同 Ollama 基线下的归一化对比。

图表保存到：

```text
/Users/augustus/Desktop/LLMBenchmark/results/mac_benchmark/charts/
```

---

## 九、与 LLMRouterBench 主项目的衔接

Qwen3.6-35B-A3B 的实测结果可以从三个层面接入 LLMRouterBench：

1. **模型能力层**
   - 通过 Ollama API 采集准确率、工具调用、thinking 和多模态结果；
   - 同一模型在不同量化上的结果要分别记录，不能混成一条模型记录。

2. **性能成本层**
   - 记录 TTFT、Prefill tok/s、Decode tok/s、峰值内存与可用上下文；
   - 这些指标可作为路由的延迟成本、内存成本和上下文能力约束；
   - 与云端 API 的 `pricing` 并列使用时，应明确本地成本口径。

3. **硬件节点层**
   - M4 Pro 与 M5 Pro 是不同硬件节点；
   - 同一模型在 M4 的 Ollama、M5 的 Ollama、M5 的 MLX 上是三个不同的执行配置；
   - 路由实验必须保存引擎、量化、上下文和硬件 fingerprint。

推荐为每次实验保存：

```json
{
  "hardware_fingerprint": {
    "chip": "Apple M5 Pro",
    "model_identifier": "Mac17,16",
    "memory_gib": 64,
    "os_version": "27.2"
  },
  "engine_fingerprint": {
    "name": "ollama",
    "version": "0.33.3",
    "port": 11435
  },
  "model_fingerprint": {
    "name": "qwen3.6:35b-a3b-q4_K_M",
    "format": "gguf",
    "quantization": "Q4_K_M",
    "digest": "..."
  }
}
```

---

## 十、执行清单（Qwen3.6-35B-A3B）

| 阶段 | 动作 | 在哪执行 | 预计耗时 |
|---|---|---|---|
| Step 0 | 核查三台机器 Ollama 版本、digest、端口和可用内存 | 本机 SSH | 10 min |
| Step 0 | 记录 3.13 Splash 运行状态与 RSS（当前实测约 22.6GiB、空闲内存约 29%），正式基线前停止或完全释放 | 3.13 | 5 min |
| Step 1 | 三台机器执行 Ollama 最小生成验证 | 三台 | 10 min |
| Step 1 | 建立 8K / 32K / 64K 上下文基线 | 三台 | 60-90 min |
| Step 1 | 重复 Decode 稳定性测试并采集内存/swap | 三台 | 45 min |
| Step 2 | 在 3.5 检查磁盘与 MLX-LM 环境 | 3.5 | 5 min |
| Step 2 | 下载 Qwen3.6 MLX 4bit（约 20.4GB，需确认） | 3.5 | 20-60 min |
| Step 2 | 最小加载与文本生成验证 | 3.5 | 10 min |
| Step 2 | MLX Prefill / Decode / 内存 / 上下文测试 | 3.5 | 60 min |
| Step 3 | 同机 Ollama 与 MLX 交叉验证 | 3.5 | 30 min |
| Step 3 | M4 Pro 与 M5 Pro 硬件对比 | 本机汇总 | 20 min |
| Step 4 | 生成 Markdown 报告和图表 | 本机 | 30 min |

**下载门槛：** 第一阶段用 Ollama 即可完成能力与统一性能基线。只有完成基线后，才决定是否下载约 20.4GB 的 MLX 权重。

**正式基线环境要求：** 3.13 当前 Splash 常驻会占用约 22.6GiB RSS，并使系统空闲内存降至约 29%，直接测试会让 Qwen3.6-35B-A3B 的内存和速度结果失真。执行 Ollama/MLX 正式测试前，应停止 Splash 或使用一台完全空闲的机器；若必须保留，则在结果中单列“有后台负载”场景，不能与空闲基线混排。

---

## 十一、第二阶段预告（生图 / 生视频）

本文档只覆盖文本模型。第二阶段计划评估：

- **Qwen-Image 20B（生图）**：需先确认适合 Apple Silicon 的量化版本和运行时；
- **Wan2.2 14B（生视频）**：需确认显存/统一内存占用、分辨率、帧数和时长设置。

评测维度需要新增：

- 图像生成：首图延迟、每秒迭代步数、峰值内存、输出质量；
- 视频生成：首帧延迟、吞吐、单视频总时长、峰值内存、磁盘 I/O。

三台机器均为 64GB 统一内存，但是否能稳定完成生成任务必须以实测为准，不能仅凭参数量推断。

---

## 十二、Gemma-4-26B-A4B 部署与路由评测引擎选择

### 12.1 结论

**建议统一用 Ollama 作为 LLMRouterBench 路由评测的主推理引擎，MLX 保留为 Mac 原生性能专项。**

这不是因为 LLMRouterBench 强制要求统一引擎。项目的数据采集器只依赖 OpenAI 兼容的 `/v1/chat/completions` 接口，底层可以是 Ollama、llama.cpp、vLLM、MLX 服务或云端 API。但路由评测需要把**模型能力差异**与**推理引擎差异**分开；如果池内模型混用 Ollama GGUF 与 MLX 量化权重，路由分数会同时包含“模型差异 + 引擎差异 + 量化差异”，不利于解释和复现。

### 12.2 实机核查结果

| 机器 | 当前 Gemma-4-26B-A4B 状态 | 是否可直接接入路由评测 |
|---|---|---|
| 192.168.3.7 M4 Pro | 有完整 MLX 4bit 权重：`~/.omlx/models/gemma-4-26b-a4b-it-4bit`，3 个 safetensors，约 14.54GiB；另有 GGUF `~/workspace/GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`，约 16GB | ❌ 尚未作为 OpenAI 兼容服务稳定暴露；Ollama 模型列表中没有 Gemma |
| 192.168.3.5 M5 Pro | Ollama 0.33.3 正常，但只有 Qwen3.6-35B-A3B 与 Qwen3.5-9B；没有 Gemma | ❌ 需要先导入或拉取 Gemma |
| 192.168.3.13 M5 Pro | 有 Ollama 与 MLX 环境，但 Splash 常驻占用约 22.6GiB，且无 Gemma | ❌ 不建议作为 Gemma 主测机 |

> 3.7 的 MLX 权重位于 oMLX 私有模型目录，不是标准 `~/Models/...` 路径。它可以被复制或重新下载到标准目录后由 MLX-LM 加载，但不能直接把 oMLX 的内部缓存当成已部署的 MLX-LM 服务。

### 12.3 推荐部署路径

#### 路径 A：Ollama 统一评测（推荐）

在 3.5 上部署 Gemma-4-26B-A4B，与 Qwen3.6-35B-A3B 使用同一个 Ollama 服务。优先尝试直接拉取官方模型：

```bash
ssh macmini-m5-jiuwen

ollama pull gemma4:26b-a4b-it-q4_K_M
ollama list
curl -fsS http://127.0.0.1:11435/v1/models | python3 -m json.tool
```

> 已核对 Ollama 官方标签，`gemma4:26b-a4b-it-q4_K_M` 当前存在。

如果 3.5 已能访问 3.7 的 GGUF 文件，也可以复制后通过 Modelfile 导入，确保量化级别明确：

```bash
# 从本机把 3.7 的 GGUF 复制到 3.5（约 16GB）
scp macmini-m4-admin:~/workspace/GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf /tmp/

ssh macmini-m5-jiuwen
mkdir -p ~/Models/gemma-4-26b-a4b
mv /tmp/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf ~/Models/gemma-4-26b-a4b/

cat > ~/Models/gemma-4-26b-a4b/Modelfile <<'MODELFILE'
FROM ./gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
PARAMETER temperature 0
PARAMETER num_ctx 32768
MODELFILE

ollama create gemma-4-26b-a4b:q4_K_M -f ~/Models/gemma-4-26b-a4b/Modelfile
ollama list
```

注意：3.5 当前磁盘可用约 489GiB，容纳 16GB GGUF 没有压力；但导入后会再生成 Ollama blob，实际占用可能接近原始文件大小，部署前应再次确认 `df -h ~`。

#### 路径 B：复用 3.7 的 MLX 权重做性能专项

如果目标是测 Mac 原生 Prefill/Decode/峰值内存，而不是统一路由能力基线，可在 3.7 或 3.5 使用 MLX-LM。3.7 当前没有 `mlx_lm.generate` CLI，因此需要先安装 MLX-LM，或将权重复制到 3.5 后运行：

```bash
# 在 3.5 上准备标准模型目录
mkdir -p ~/Models
scp -r macmini-m4-admin:~/.omlx/models/gemma-4-26b-a4b-it-4bit \
  ~/Models/gemma-4-26b-a4b-it-4bit
```

然后在 3.5 启动独立服务，避免与 Ollama 的 11435 冲突：

```bash
/opt/homebrew/opt/mlx-lm/libexec/bin/mlx_lm.server \
  --model ~/Models/gemma-4-26b-a4b-it-4bit \
  --host 127.0.0.1 \
  --port 8081 \
  --temp 0.0 \
  --max-tokens 512
```

MLX 路径只用于性能专项或引擎对照；不建议把它与 Ollama 的 Gemma 结果合并成同一条路由模型记录。

### 12.4 LLMRouterBench 配置建议

如果采用 Ollama 统一路线，应新增专用配置，不要直接改公共模板。**关键限制：Ollama 的 OpenAI 兼容接口 `/v1/chat/completions` 支持 `max_tokens`，但不支持通过 `extra_body.options.num_ctx` 设置上下文。** 因此上下文长度必须在 Modelfile 或 Ollama 原生 `/api/chat` 请求中固定，不能只写在 LLMRouterBench 的 YAML 里。

先用标准标签拉取模型，并通过临时 Modelfile 固化评测参数：

```bash
ssh macmini-m5-jiuwen

ollama pull gemma4:26b-a4b-it-q4_K_M

cat > /tmp/Modelfile.gemma4-bench <<'MODELFILE'
FROM gemma4:26b-a4b-it-q4_K_M
PARAMETER num_ctx 32768
PARAMETER temperature 0
PARAMETER num_predict 512
MODELFILE

cat > /tmp/Modelfile.qwen36-bench <<'MODELFILE'
FROM qwen3.6:35b-a3b-q4_K_M
PARAMETER num_ctx 32768
PARAMETER temperature 0
PARAMETER num_predict 512
MODELFILE

ollama create gemma4-26b-a4b-it-q4_K_M-ctx32k \
  -f /tmp/Modelfile.gemma4-bench
ollama create qwen3.6-35b-a3b-q4_K_M-ctx32k \
  -f /tmp/Modelfile.qwen36-bench

ollama show gemma4-26b-a4b-it-q4_K_M-ctx32k
ollama show qwen3.6-35b-a3b-q4_K_M-ctx32k
```

然后配置 LLMRouterBench：

```yaml
models:
  - name: Gemma-4-26B-A4B-Ollama-Q4KM-32K-M5
    api_model_name: gemma4-26b-a4b-it-q4_K_M-ctx32k
    base_url: http://127.0.0.1:11435/v1
    api_key: ollama
    generator_type: direct
    temperature: 0
    top_p: 1.0
    timeout: 1800
    extra_body:
      max_tokens: 512

  - name: Qwen3.6-35B-A3B-Ollama-Q4KM-32K-M5
    api_model_name: qwen3.6-35b-a3b-q4_K_M-ctx32k
    base_url: http://127.0.0.1:11435/v1
    api_key: ollama
    generator_type: direct
    temperature: 0
    top_p: 1.0
    timeout: 1800
    extra_body:
      max_tokens: 512
```

> `max_tokens` 是 OpenAI 兼容请求字段，可以放在 `extra_body` 中透传；`num_ctx` 不是该接口的兼容字段。若需要按数据集动态切换上下文，应使用 Ollama 原生 `/api/chat` 适配器，或为每个上下文长度创建独立的 Ollama 模型别名。

建议：

1. `name` 中写明模型、引擎、量化、上下文和机器，例如 `Gemma-4-26B-A4B-Ollama-Q4KM-32K-M5`，避免不同执行配置在结果目录中重名。
2. 路由主实验只纳入同一引擎、同一量化级别的模型；MLX 结果放到独立的性能报告或消融实验。
3. 同一台机器上的模型统一走一个 Ollama 服务；不同机器仍使用各自的 `base_url`，但 API 协议、模型量化等级和请求参数必须一致。
4. 如果未来必须混用 Ollama 与 MLX，则至少在结果中保留 `engine`、`quantization`、`base_url`、`hardware` 四个字段，并单独做引擎效应分析。
5. 配置中的 `api_key: ollama` 只是本地占位值，不要写入真实云端密钥。

**路由池的模型命名要区分“模型身份”和“执行节点”。** 同一模型在 3.5 与 3.7 上运行应视为两个候选执行节点，例如：

```text
Gemma-4-26B-A4B-Ollama-Q4KM-32K-M5
Gemma-4-26B-A4B-Ollama-Q4KM-32K-M4
Qwen3.6-35B-A3B-Ollama-Q4KM-32K-M5
Qwen3.6-35B-A3B-Ollama-Q4KM-32K-M4
```

这样路由训练才能同时学习“选哪个模型”和“选哪个执行节点”。如果只关心模型能力差异，则应固定同一硬件节点，避免把机器差异误学成模型偏好。

### 12.5 为什么不建议直接用 3.7 的 oMLX 缓存

- oMLX 的模型目录包含其私有元数据（`.msc`、`.mv`、`configuration.json`），不能保证标准 MLX-LM 能直接加载。
- 3.7 当前没有发现全局 `mlx_lm` CLI，说明该模型可能由 oMLX、EXO 或 LM Studio 内部运行时管理，而不是一个可复用的评测服务。
- 3.7 同时运行 EXO 与 LM Studio 相关进程，正式路由评测时会产生资源竞争与端口管理问题。
- 如果只是复用权重，最稳妥的方法是把 safetensors 复制到标准目录并在 3.5 用 MLX-LM 显式启动；不要依赖 GUI 应用隐式拉起模型。

### 12.6 推荐执行顺序

```text
1. 在 3.5 拉取官方 `gemma4:26b-a4b-it-q4_K_M`，必要时导入 3.7 的 GGUF
2. 分别通过 Modelfile 为 Gemma 与 Qwen 固化 `num_ctx=32768`、`temperature=0`、`num_predict=512` 的评测别名
3. 用 `/v1/models` 和一次最小 `/v1/chat/completions` 请求验证 OpenAI 兼容接口
4. 把 Gemma 与 Qwen3.6 放在同一 Ollama 服务、同一数据集子集上跑路由基线
5. 记录每个 run 的模型 digest、量化、Ollama 版本和机器 fingerprint
6. 另开 MLX 专项：在 3.5 用标准 MLX-LM 加载 Gemma 4bit，测 Prefill/Decode/Peak memory
7. 最后再决定是否把 MLX 结果纳入路由池；默认不混入主基线
```

**一句话结论：** 路由评测优先统一 Ollama；3.7 已有的 Gemma MLX 权重应作为性能专项或备用资产，而不是直接替代统一的 Ollama 部署。

### 12.7 首次部署详细操作手册

Gemma-4-26B-A4B 在共享 M5 Pro 机器上的下载、加载、验收、卸载与共享使用规范，见：

- [Gemma-4-26B-A4B Ollama 首次部署操作手册](./Gemma-4-26B-A4B-Ollama部署操作手册.md)

核心原则：下载阶段影响相对可控；首次加载、推理与模型驻留会显著占用统一内存和 GPU，必须在独占窗口进行，并在测试后执行 `ollama stop`。

### 12.8 模型下载与网络瓶颈处理

Gemma 的下载失败与低速问题，不应简单归因于模型过大。当前实测表明，3.5 和 3.7 都通过 Wi-Fi `en1` 接入 `192.168.3.0/24`，两台机器的 Ethernet `en0` 均为 `inactive`；3.5 的 Wi-Fi 虽为 802.11ac 5GHz 且协商速率约 585 Mbps，但互联网实际下行在多次测试中只有约 2–17.5 Mbps，波动很大。

下载时按以下顺序处理：

1. **优先接有线 Ethernet**：两台机器接入同一千兆/2.5GbE 交换机，确认 `en0` 为 `active` 且默认路由走 `en0`；然后用 `networkQuality` 和局域网大文件复制复测。当前无线链路上已观测到约 20% 丢包和平均约 788ms RTT，必须先排除无线质量问题。
2. **局域网优先复用已验证 GGUF**：若 3.7 的完整文件满足目标版本，先复制并核对大小和 SHA256，再用 `ollama create` 导入，速度通常比公网稳定。
3. **公网可尝试 ModelScope + aria2 多连接**：ModelScope 支持 Range 请求，可用 `aria2c -x 4 -s 4 -k 1M -c`；但多连接不能突破出口总带宽，共享网络下应限制并发。
4. **Hugging Face 不是当前稳定来源**：3.5 多次测试出现 `Connection reset by peer`，应在目标时段重新测速，不要仅凭一次成功或失败判断。
5. **保留 Ollama 断点数据**：不要删除 `~/.ollama` 或手动修改 `-partial` 文件；重新执行相同 `pull` 以复用已有分块。

**评测口径要求：**

- Ollama 官方 tag、3.7 现有 Unsloth GGUF、ModelScope 最新 GGUF 可能是不同文件，不能混为同一模型版本；
- 每个结果至少记录模型名称、来源、文件大小、SHA256、量化方式、Ollama 版本与 `num_ctx`；
- 网络下载耗时不计入 Prefill/Decode 性能指标；
- 共享机器下载应避开办公高峰，正式加载和推理仍需独占窗口。

详细命令、实测数据和来源对比见：

- [Gemma-4-26B-A4B Ollama 部署操作手册：13.8–13.9 节](./Gemma-4-26B-A4B-Ollama部署操作手册.md)


