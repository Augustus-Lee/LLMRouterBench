# Splash 四模型评测可行性分析

> 核查日期：2026-11-24（Asia/Shanghai）  
> 目标机器：`192.168.3.13` / `SynCo-MacMini` / Apple M5 Pro / 64GB 统一内存 / macOS 27.0  
> 目标框架：Splash `1.5.0`  
> 目标模型：`Qwen3.8-27B`、`Qwen3.6-35B-A3B`、`Gemma-4-26B-A4B`、`Qwen3.8-Flash-Next`  
> 结论范围：单模型、单实例、Splash 推理引擎。本文不把 MLX、Ollama、llama.cpp 的数值混入 Splash 结果。

## 1. 结论摘要

四款模型不能简单归为“都能测”或“都不能测”。必须先分开回答四个问题：

1. **Splash 是否认识这个模型架构？**
2. **当前机器上是否有完整、符合 Splash 要求的模型资产？**
3. **64GB 内存能否稳定完成预期上下文的评测？**
4. **达到的是“可运行”，还是“速度上限/质量上限”？**

| 模型 | Splash 架构支持 | 当前 3.13 资产 | 64GB 单模型评测 | 能否发挥最大性能 | 总结论 |
|---|---|---|---|---|---|
| Qwen3.8-27B | ✅ 原生支持 | ✅ Q4_K_M、mmproj、DFlash2 完整 | ✅ 推荐 | ⚠️ 可发挥 Q4 下的较优速度，不能同时宣称质量上限 | **首选正式基线** |
| Qwen3.6-35B-A3B | ✅ 原生支持 `qwen35moe` | ❌ 仅有 Ollama blob，不是 Splash 标准资产 | ✅ 补齐资产后可 | ⚠️ 可发挥 MoE 速度优势，需独占机器 | **第二优先，先补资产** |
| Gemma-4-26B-A4B | ❌ 1.5.0 不支持 `gemma4` | ❌ 3.13 无；3.7 有 GGUF/MLX | 硬件上可运行，但 Splash 不可 | ❌ 不能在 Splash 判断 | **当前阻塞：等 Splash 支持** |
| Qwen3.8-Flash-Next | ✅ 原生支持 `qwen4exp` | ❌ 3.13 无 | ⚠️ 只能低比特 + SSD 流式实验 | ❌ 不能代表速度或质量上限 | **不作为正式基线** |

### 一句话判断

- **最适合马上做正式 Splash 基线的是 Qwen3.8-27B。**
- **Qwen3.6-35B-A3B 原生支持，但当前只有 Ollama 权重；需要补 Splash 标准 GGUF、mmproj 和 DFlash2。**
- **Gemma-4-26B-A4B 不是机器问题，而是 Splash 1.5.0 架构不支持，当前不能评测。**
- **Qwen3.8-Flash-Next 可以被 Splash 识别，但 64GB 需要把 n-gram/PLE 表放到 SSD 流式读取；“能跑”不等于“能发挥最大性能”。**

## 2. 测试环境与当前状态

### 2.1 机器配置

| 项目 | 实测值 |
|---|---|
| 主机名 | `SynCo-MacMini` |
| 地址 | `192.168.3.13` |
| 芯片 | Apple M5 Pro |
| CPU | 18 核（6 个性能核 + 12 个能效核） |
| GPU | 20 核，Metal 4 |
| 统一内存 | 64GB |
| 系统 | macOS 27.0 |
| 数据盘 | 926GiB，已用约 802GiB，可用约 78GiB（92% 使用率） |
| Splash | `/opt/homebrew/bin/splash`，版本 1.5.0 |

### 2.2 本次复核时的运行状态

本次只读复核时：

- `18080` 上没有 Splash 监听；
- 没有发现 Splash 推理进程；
- NVIDIA PAIR 代理仍监听 `8080`；
- PAIR 拉起的 Ollama 仍监听 `127.0.0.1:11435`；
- Ollama 当前已加载 `qwen3.6:35b-a3b-q4_K_M`，进程 RSS 约 29GiB；
- 系统显示空闲内存比例约 56%，但 swap 历史计数较高；
- `memory_pressure` 的 swap 使用约 635MiB / 2GiB。

这说明：

1. **仓库中历史的“Splash 正在 18080 常驻”描述已经过期。**
2. 当前机器并不处于干净的 benchmark 状态。
3. 正式测试前必须重新确认进程、内存、swap、端口和磁盘，不能照搬旧日志。
4. 未经协调不能停止 NVIDIA PAIR/Ollama，因为这是共享机器上的其他服务。

### 2.3 正式测试的环境底线

正式性能结果至少应满足：

- 单模型、单实例、单请求；
- 停止或错开其他大模型推理；
- 固定上下文、量化、KV 类型、采样参数和 reasoning effort；
- 记录预热与正式重复次数；
- 记录 TTFT、Prefill、Decode、峰值 RSS、wired memory、swap、SSD 读；
- 记录模型仓库、revision、文件字节数、Splash 版本和启动命令；
- 不把旧的交互日志当作正式基线。

## 3. Splash 1.5.0 的能力与边界

### 3.1 原生支持的 GGUF 文本架构

从本机 Splash 1.5.0 的安装源码核实，`ARCHITECTURES` 仅包含：

```text
qwen35
qwen35moe
qwen4exp
flint
```

模型族映射为：

```text
Qwen3.8-27B
Qwen3.6-35B-A3B
Qwen3.5-122B-A10B
Qwen3.8-Flash-Next
Flint
```

其中：

- `Qwen3.8-27B` 映射到 `qwen35`；
- `Qwen3.6-35B-A3B` 映射到 `qwen35moe`；
- `Qwen3.8-Flash-Next` 映射到 `qwen4exp`；
- `Gemma-4-26B-A4B` 的 `gemma4` 不在架构表内。

对应 DFlash2：

```text
Qwen3.8-27B          -> incoai/Qwen3.8-27B-DFlash2
Qwen3.6-35B-A3B      -> incoai/Qwen3.6-35B-A3B-DFlash2
Qwen3.8-Flash-Next   -> incoai/Qwen3.8-Flash-Next-DFlash2
```

### 3.2 与评测有关的关键参数

| 参数 | 作用 | 对评测的影响 |
|---|---|---|
| `--max-memory` | 限制 Metal 内存预算 | 太低会降速或加载失败，太高会挤压系统 |
| `--max-context` | 设置上下文 token 上限 | 决定 KV、state 和内存压力 |
| `--kv-format int8\|bf16` | KV Cache 类型 | INT8 更省内存，BF16 更利于质量上限 |
| `--state-format fp32\|bf16` | 循环状态精度 | 影响长上下文内存和数值稳定性 |
| `--ple-memory` | Flash-Next 的 PLE/n-gram 表 RAM 缓存 | 不设或设小会从 SSD 流式读取，显著影响速度 |
| `--expert-memory` | MoE routed experts 的 RAM 槽位 | 不足时按 token 从 SSD 读专家，速度下降 |
| `--sparse-kv-memory` | 块稀疏 KV 的 RAM 缓存 | Flash-Next 长上下文相关 |
| `--draft-model` | 指定 DFlash2 draft | 投机解码影响 Decode 速度 |
| `--verify-rows auto` | 自适应 MoE draft 校验 | 更快，但贪心输出可能不完全可复现 |
| `--verify-rows all` | 校验所有 draft 行 | 更可复现，但更慢 |
| `--decode-share` | 多请求时 prefill/decode 的时间分配 | 单请求基准应固定并说明 |
| `--queue-size` | 并发准入数 | 正式单请求测试不应让并发污染结果 |
| `--idle-release` | 空闲后释放权重 | 适合共享机器，但复测会重新加载 |

### 3.3 三个容易混淆的“支持”

**架构支持**不等于**仓库资产完整**：模型架构在 `ARCHITECTURES` 中，仍可能缺少合适的 GGUF、mmproj 或 tokenizer。

**能加载**不等于**有足够内存稳定评测**：64GB 下可以使用 mmap/SSD 流式读取，但这会把瓶颈转移到 SSD 带宽和缓存命中率。

**能出 tok/s**不等于**发挥模型最大性能**：低比特量化、短上下文、缺少 DFlash2、INT8 KV、SSD 流式读取都会改变结果。

## 4. 模型逐一分析

### 4.1 Qwen3.8-27B

#### 结论

**Splash 可行性最高，是当前 3.13 上最适合先做的正式 Q4 基线。**  
可以在 64GB 单模型独占条件下完成中等上下文评测；但不能把“Q4 下的较优速度”说成“模型质量上限”。

#### 架构与资产

- Splash 架构：`qwen35`；
- DFlash2：`incoai/Qwen3.8-27B-DFlash2`；
- 当前 GGUF：`Qwen3.8-27B-UD-Q4_K_M.gguf`，约 15.33GiB；
- 视觉投影：`mmproj-BF16.gguf`，约 887MiB；
- DFlash2 draft：约 3.58GiB；
- 同目录其他 Q5/Q6/Q8 文件多为 135～136 字节的 LFS pointer，不是完整权重。

#### 上下文与内存判断

模型原生最大位置编码为 262,144 tokens。使用混合注意力，只有部分层是全注意力层。按配置粗略估算，INT8 KV 在 128K 时约 4GiB，在 256K 时约 8GiB；BF16 KV 则约为 8GiB 和 16GiB。

这只是 KV 的理论下限，不含：

- 权重；
- DFlash2 draft；
- vision projector；
- Metal 临时缓冲区；
- ANE 程序；
- 操作系统和其他进程；
- allocator 碎片。

因此：

- **128K 是现实起点**；
- **256K 需要单独探测，不应默认成功**；
- 若同时保留 27GB 级 Ollama 进程，正式长上下文测试不可信。

#### 推荐启动配置

正式基线建议先使用：

```bash
/opt/homebrew/bin/splash serve \
  --model-dir /Users/synco/Models/splash/Qwen3.8-27B-GGUF \
  --mmproj /Users/synco/Models/splash/Qwen3.8-27B-GGUF/mmproj-BF16.gguf \
  --draft-model /Users/synco/Models/splash/Qwen3.8-27B-DFlash2 \
  --port 18080 \
  --max-context 128K \
  --max-memory 40G \
  --kv-format int8 \
  --state-format bf16 \
  --default-reasoning-effort medium \
  --verify-rows all \
  --queue-size 1 \
  --request-timeout 30m \
  --idle-release 10m
```

说明：

- `--verify-rows all` 用于正式可复现实验；若要测速度上限，可另做 `auto` 对照，但不能混在同一张表里。
- `--max-memory 40G` 是保守起点，不是越大越好；需要根据实际峰值调整。
- 当前目录中可用的完整权重只有 Q4_K_M，所以“正式 Q4 基线”成立，“最高质量评测”不成立。
- 历史日志显示曾以 `--max-context 131072` 运行，但日志中的用户路径、视觉状态和当前服务状态存在不一致，不能直接当作本次干净基线。

#### 能否发挥最大性能

- **速度上限方面**：接近可发挥 Q4 + DFlash2 + INT8 KV 的较好状态，前提是独占机器。
- **质量上限方面**：不能。Q4_K_M 不是最高质量量化，128K 也不是完整 256K 原生窗口。
- **视觉能力方面**：资产存在 mmproj，但历史日志曾显示 `text only`；正式视觉评测前必须用实际图片请求和 `/status` 重新确认。

### 4.2 Qwen3.6-35B-A3B

#### 结论

**Splash 原生支持，64GB 单模型可评测；但当前 3.13 上只有 Ollama blob，不能直接当作 Splash 正式资产。**  
补齐标准 GGUF、mmproj 和 DFlash2 后，可作为第二优先的 MoE 正式基线。

#### 架构与资产

- Splash 架构：`qwen35moe`；
- DFlash2：`incoai/Qwen3.6-35B-A3B-DFlash2`；
- 官方 Splash 帮助中直接给出：

```bash
splash serve --model unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M --max-context 128K
```

当前 3.13：

- Ollama 模型：`qwen3.6:35b-a3b-q4_K_M`；
- blob 约 22.3GiB；
- 参数约 35.95B；
- 架构 `qwen35moe`；
- 训练上下文 262,144；
- 当前 Ollama 运行上下文 65,536；
- manifest 中只有模型、license、params 三类层，没有 Splash 所需的独立 tokenizer/mmproj/DFlash2 资产结构。

因此：

- **Ollama 能加载，不代表 Splash 能直接使用这个 blob。**
- 不能把 Ollama 的 TTFT/Decode 填进 Splash 结果表。

#### 建议补齐的资产

推荐独立目录，例如：

```text
/Users/synco/Models/splash/Qwen3.6-35B-A3B-GGUF/
  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  mmproj-BF16.gguf

/Users/synco/Models/splash/Qwen3.6-35B-A3B-DFlash2/
  model.safetensors
  config.json
  ...
```

已核实的上游文件大小：

```text
Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   22,134,528,992 bytes ≈ 20.62GiB
mmproj-BF16.gguf                    902,822,624 bytes ≈ 861MiB
```

下载这三个主要部分约需 24GiB 级空间。3.13 当前可用约 78GiB，空间够，但下载前仍应确认不会挤压系统盘和缓存。

#### 上下文与内存判断

该模型原生上下文同为 262,144；有 40 层，其中 10 层为全注意力。按配置粗略估算：

- INT8 KV：128K 约 1.25GiB，256K 约 2.5GiB；
- BF16 KV：128K 约 2.5GiB，256K 约 5GiB。

KV 不是主要瓶颈，**22.3GiB 的权重、MoE expert 常驻、DFlash2、vision projector、Metal 缓冲以及系统其他进程**才是主要内存来源。

在 64GB 上：

- 32K/64K/128K 都有现实可行性；
- 256K 需要单独探测；
- 与当前 27GB 级 Ollama 进程同时运行不适合正式测试；
- 若启用 `--expert-memory` 限制，部分专家从 SSD 读取会显著改变速度，必须单独标注。

#### 推荐启动配置

```bash
/opt/homebrew/bin/splash serve \
  --model-dir /Users/synco/Models/splash/Qwen3.6-35B-A3B-GGUF \
  --mmproj /Users/synco/Models/splash/Qwen3.6-35B-A3B-GGUF/mmproj-BF16.gguf \
  --draft-model /Users/synco/Models/splash/Qwen3.6-35B-A3B-DFlash2 \
  --port 18081 \
  --max-context 128K \
  --max-memory 42G \
  --kv-format int8 \
  --state-format bf16 \
  --default-reasoning-effort medium \
  --verify-rows all \
  --queue-size 1 \
  --request-timeout 30m \
  --idle-release 10m
```

#### 能否发挥最大性能

- **速度上限方面**：有条件接近。35B 总参数但只有约 3B 激活，MoE 是它的核心优势；需要 DFlash2、足够大的 expert 常驻内存、INT8 KV 和独占机器。
- **质量上限方面**：Q4_K_M 不是最高质量；原生 256K 也不等于 64GB 下所有 256K 场景都稳定。
- **当前阻塞项**：先补齐 Splash 标准资产，否则无法得到有效的 Splash 数据。

### 4.3 Gemma-4-26B-A4B

#### 结论

**硬件上能运行，但当前 Splash 1.5.0 不能正式评测。**  
这不是“权重不够小”或“机器不够快”，而是框架架构不支持。

#### 为什么不支持

Splash 1.5.0 的架构集合只有：

```text
qwen35
qwen35moe
qwen4exp
flint
```

没有 `gemma4`。模型族列表中也没有 Gemma。

3.7 机器上已有的 GGUF 文件头可直接确认：

```text
general.architecture = gemma4
```

模型配置：

```text
architectures: Gemma4ForConditionalGeneration
model_type: gemma4
text_config.model_type: gemma4_text
max_position_embeddings: 262144
sliding_window: 1024
num_hidden_layers: 30
num_experts: 128
```

因此：

- 把 3.7 的 GGUF 复制到 3.13 **不能解决** Splash 兼容性问题；
- 3.7 的 MLX 4bit 结果不能替代 Splash 结果；
- 不能通过“改模型名”绕过架构检查；
- 若强行用其他运行时，只能作为另一个引擎实验，不能与 Splash 数据同表。

#### 当前资产

- 3.13：未发现完整 Gemma-4-26B-A4B 权重；
- 3.7：有 GGUF `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`，约 15.7GiB；
- 3.7：有 MLX 4bit 权重，约 15.6GB safetensors；
- 3.7 的 Ollama manifest 中也未见 Gemma 条目。

#### 可选后续路径

1. **等待 Splash 增加 `gemma4` 支持或提供定制构建**；这是保持同框架结论的唯一干净路径。
2. 在 MLX/llama.cpp/Ollama 上单独评测，明确标注“非 Splash”。
3. 如果只是做硬件能力摸底，可在 3.7 的 MLX 上做，但结论只能回答“该模型在该引擎/该机器”的性能，不能回答“Splash 下”的性能。

#### 能否发挥最大性能

在 Splash 下：**无法判断，也无法优化**。  
在 MLX 或其他引擎下：可以另做，但不能与本文的 Splash 结果横向混排。

### 4.4 Qwen3.8-Flash-Next

#### 结论

**Splash 代码层原生支持，但 3.13 的 64GB 内存和约 78GiB 可用磁盘只允许低比特、SSD 流式读取的实验，不能作为正式速度/质量上限基线。**

#### 模型规模不能按“6B 激活”估算

官方模型说明：

```text
125B total parameters
6B activated
+ 51B n-gram embedding
+ 4B MTP
```

因此：

- 计算激活量约 6B，只能说明算力路径较轻；
- **权重总量仍然巨大**；
- n-gram 表是另一个独立的大资产；
- 不能因此认为 64GB 足够完整驻留。

#### Splash 支持情况

Splash 1.5.0：

- 支持 `qwen4exp`；
- 有 `Qwen3.8-Flash-Next` 模型族；
- 有 `incoai/Qwen3.8-Flash-Next-DFlash2`；
- 提供 `--ple-memory`、`--expert-memory`、`--sparse-kv-memory`，用于把部分大表/专家/稀疏 KV 放到 SSD 读取。

Splash 自己的推荐表还列出了：

```text
ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0
建议最小内存：64GB
下载体积：约 69GB
```

这说明“64GB 机器可以启动”是有条件的：**必须把 n-gram/PLE 表放在 SSD 上按需读取，不能完整驻留内存。**

#### 量化与磁盘边界

已核实的 Unsloth GGUF 总大小（十进制 GB）：

| 量化 | 总大小 |
|---|---:|
| UD-IQ1_S | 66.38GB |
| UD-IQ1_M | 68.01GB |
| UD-IQ2_XS | 75.83GB |
| UD-Q2_K_XL | 79.58GB |
| UD-IQ3_XXS | 83.63GB |
| UD-Q3_K_XL | 89.46GB |
| UD-IQ4_XS | 93.67GB |
| UD-Q4_K_XL | 111.33GB |

ISTA-DASLab 的 GSQ-RCO 版本提供更明确的拆分：

| 版本 | 第一分片需常驻 | 第二分片 n-gram 表可放盘 | 总下载 |
|---|---:|---:|---:|
| Q2_0 | 37.6GB | 28.8GB | 66.4GB |
| IQ2_XS | 39.2GB | 28.8GB | 68.0GB |
| IQ3_XXS | 47.0GB | 28.8GB | 75.8GB |
| IQ3_S | 54.8GB | 28.8GB | 83.6GB |

3.13 当前可用约 78GiB：

- Q2_0 总量约 62GiB，理论能放下，但下载后只剩十几 GiB；
- IQ2_XS 总量约 63GiB，也接近把剩余磁盘吃满；
- IQ3_XXS/IQ3_S 不适合当前磁盘余量；
- 还需为 mmproj、KV、缓存、日志和系统预留空间。

#### 速度与质量判断

ISTA 的测量是在 llama.cpp 上得到的，不是 Splash，不能直接当作本机 Splash 结果；但它说明了关键趋势：

- Q2_0 的 Decode 比 IQ2_XS 更稳定，因为避免了查表开销；
- n-gram 表放 SSD 会增加 paging 成本；
- 低比特会牺牲质量；
- “能跑”与“发挥最大性能”不是一回事。

在 Splash 上应预期：

- Prefill 和 Decode 受 SSD 读带宽、n-gram 命中率、expert/PLE 缓存大小影响；
- 短上下文、低并发可能能完成实验；
- 长上下文和多轮请求会放大 paging 和 KV 压力；
- 结果不适合作为模型上限或跨机器正式基线。

#### 推荐实验配置（仅实验）

前提：先协调独占机器、清理足够磁盘、停止其他大模型服务。

```bash
/opt/homebrew/bin/splash serve \
  --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 \
  --port 18082 \
  --max-context 32K \
  --max-memory 48G \
  --ple-memory 2G \
  --expert-memory 16G \
  --sparse-kv-memory 1200M \
  --kv-format int8 \
  --state-format bf16 \
  --queue-size 1 \
  --request-timeout 30m
```

说明：

- 这是**实验配置**，不是“最大性能配置”；
- `--ple-memory 2G` 会让大部分 n-gram 表从 SSD 读；
- `--expert-memory 16G` 也可能让部分专家走 SSD；
- 32K 只是控制风险的起点，不代表模型能力上限；
- 应先做短请求 smoke test，再逐步增加上下文；
- 任何 OOM、swap 激增、超时或明显卡顿都应记录为失败或降级结果。

#### 能否发挥最大性能

**不能。**  
在 3.13 上主要受限于：

- 64GB 统一内存无法完整驻留；
- 78GiB 可用磁盘不足；
- SSD 流式读取带来延迟和抖动；
- 低比特量化影响质量；
- 共享机器上的其他服务会污染结果。

若要做正式 Q2/Q3/Q4 评测，至少需要更大的统一内存和更充裕的磁盘；具体门槛应按实际量化、上下文和 KV 配置验证，不能只看参数量。

## 5. 四模型对比矩阵

| 维度 | Qwen3.8-27B | Qwen3.6-35B-A3B | Gemma-4-26B-A4B | Qwen3.8-Flash-Next |
|---|---|---|---|---|
| Splash 架构 | `qwen35` | `qwen35moe` | `gemma4` ❌ | `qwen4exp` |
| DFlash2 | ✅ | ✅ | ❌ | ✅ |
| 3.13 完整资产 | ✅ Q4 + mmproj + draft | ❌ 仅 Ollama blob | ❌ | ❌ |
| 权重规模 | 约 15.3GiB Q4 | 约 20.6GiB Q4 | 约 15.7GiB Q4（3.7） | 最小约 66GB 级 |
| 64GB 单模型 | ✅ | ✅ | 硬件可 | ⚠️ 仅流式 |
| 正式 Splash 基线 | ✅ 首选 | ⚠️ 补资产后 | ❌ | ❌ |
| 速度上限 | Q4 下较好 | MoE + DFlash2 有潜力 | 无法判断 | 不能代表 |
| 质量上限 | Q4/128K 不是上限 | Q4/128K 不是上限 | 无法判断 | 低比特/短上下文不是上限 |
| 主要风险 | 共享负载、视觉状态 | 资产不规范、SSD expert | 框架不支持 | 内存/磁盘/SSD 带宽 |

### 5.1 “能否发挥最大性能”的统一口径

本文不建议使用模糊的“能发挥最大性能”。应拆成：

**速度上限**

- 是否使用 DFlash2；
- 是否使用 INT8 KV；
- 是否启用 `--verify-rows auto`；
- 是否所有 experts 都在内存；
- 是否所有 PLE 表都在内存；
- 上下文是否足够短；
- 是否独占 GPU/ANE/内存带宽；
- 是否只有单请求。

**质量上限**

- 是否使用最高可用量化；
- 是否使用 BF16 KV；
- 是否使用完整原生上下文；
- 是否启用视觉 tower；
- 是否使用可复现的 `--verify-rows all`；
- 是否避免低比特 n-gram/expert 压缩。

这两个目标往往冲突：

- 更大上下文 → 更多 KV/state → 更慢；
- BF16 KV → 更高质量 → 更多内存；
- 更高量化 → 更高质量 → 更大权重；
- `verify-rows all` → 更可复现 → 更慢；
- SSD 流式 → 能装下更大模型 → 速度显著下降。

## 6. 推荐评测顺序与启动配置

### 6.1 推荐顺序

```text
第 1 步：Qwen3.8-27B
  Q4_K_M + DFlash2 + mmproj + INT8 KV
  先 8K/32K/64K，再 128K
  目标：建立当前机器的 Splash 基线

第 2 步：Qwen3.6-35B-A3B
  先补齐标准 GGUF + mmproj + DFlash2
  再按 8K/32K/64K/128K 测
  目标：验证 MoE 的 prefill/decode 优势

第 3 步：Gemma-4-26B-A4B
  等 Splash 增加 gemma4 支持
  目标：保持同框架可比性

第 4 步：Qwen3.8-Flash-Next
  仅在独立维护窗口做低比特/短上下文实验
  目标：验证能否加载和基本吞吐，不作为正式上限
```

### 6.2 统一评测上下文

建议至少覆盖：

```text
1K
8K
32K
64K
128K
```

Qwen3.8-27B 和 Qwen3.6-35B-A3B 可额外做 256K 探测，但必须：

- 单模型独占；
- 记录是否发生 swap；
- 记录是否发生 SSD 流式读取；
- 失败也作为结果记录，不得强行宣称支持。

### 6.3 端口规划

| 用途 | 建议端口 |
|---|---|
| NVIDIA PAIR | 8080（现有，勿冲突） |
| Ollama | 11435（现有） |
| Qwen3.8-27B Splash | 18080 |
| Qwen3.6-35B-A3B Splash | 18081 |
| Qwen3.8-Flash-Next 实验 | 18082 |

## 7. 性能评测方法

### 7.1 每次只测一个变量

推荐分层：

```text
A. 加载/可用性
B. 固定上下文下的吞吐
C. 上下文扩展
D. 量化和 KV 类型对照
E. 可复现性对照（verify-rows）
```

不要在同一次运行中同时改量化、上下文、KV 和 draft 设置。

### 7.2 必须记录的指标

| 指标 | 说明 |
|---|---|
| TTFT | 从请求发出到首个 token |
| Prefill tok/s | 输入 token / prefill 时间 |
| Decode tok/s | 输出 token / decode 时间 |
| 总延迟 | 端到端耗时 |
| 峰值 RSS | Splash 进程常驻内存 |
| Wired memory | 不可换出的内存 |
| Swap | 测试前后及测试中的增长 |
| Metal 内存 | 若 `/status` 或日志提供 |
| SSD 读 | 是否发生 PLE/expert/KV 流式读取 |
| 失败率 | OOM、超时、504、崩溃 |
| 加载时间 | 权重、draft、mmproj 的准备时间 |

### 7.3 请求设计

对每个上下文长度：

- 固定输入 token 数；
- 固定最大输出 token 数；
- 固定 temperature、top_p、top_k；
- 固定 reasoning effort；
- 关闭工具调用和联网；
- 使用同一组 prompt；
- 预热 1 次，正式重复至少 3 次；
- 记录中位数、最小值、最大值和标准差。

正式结果不应使用交互式聊天日志中的零散 tok/s。

### 7.4 可复现性

Splash 的 `--verify-rows auto` 对 MoE 更快，但贪心输出可能随运行变化。正式质量/可复现实验建议：

```text
--verify-rows all
```

速度实验可另做：

```text
--verify-rows auto
```

两者必须分开报告。

## 8. 最大性能边界

### 8.1 Qwen3.8-27B

**可达到：**

- Q4_K_M 下的较优 Splash 速度；
- 128K 上下文内的稳定评测；
- DFlash2 + INT8 KV 的内存效率。

**不能宣称：**

- 最高质量量化；
- 所有 256K 场景稳定；
- 视觉能力已经验证；
- 与其他引擎的结果可直接横比。

### 8.2 Qwen3.6-35B-A3B

**可达到：**

- MoE 的较高 prefill/decode 潜力；
- 128K 上下文内的稳定评测；
- DFlash2 下的投机解码收益。

**不能宣称：**

- Q4_K_M 是质量上限；
- Ollama 的性能等于 Splash；
- 当前 Ollama blob 可以直接作为 Splash 资产；
- 256K 在 64GB 上必然稳定。

### 8.3 Gemma-4-26B-A4B

**Splash 下：**

- 不能加载；
- 不能评测；
- 不能判断最大性能。

**其他引擎下：**

- 可以评测，但结果只能归属于对应引擎；
- 不能与 Splash 数据混表。

### 8.4 Qwen3.8-Flash-Next

**当前 3.13 可达到：**

- 低比特 + SSD 流式读取下的短上下文实验；
- 验证 Splash 是否能完成加载和基本请求。

**不能达到：**

- 完整驻留内存；
- 速度上限；
- 质量上限；
- 长上下文稳定基线；
- 与其他 20B～35B 模型同口径对比。

## 9. 风险、阻塞项与后续动作

### 9.1 阻塞项

| 阻塞项 | 影响 | 处理 |
|---|---|---|
| 3.13 共享机器上有 PAIR/Ollama | 内存、GPU、带宽污染 | 协调独占窗口；不能擅自停止 |
| Qwen3.6 只有 Ollama blob | 无法做 Splash 正式评测 | 补标准 GGUF + mmproj + DFlash2 |
| Gemma 架构不支持 | 无法启动 | 等 Splash 更新或改用其他引擎单列 |
| Flash-Next 磁盘不足 | 无法下载/缓存 | 清理磁盘或换机器 |
| 历史 Splash 日志过期 | 误把旧结果当当前基线 | 重新启动、重新验证、重新测量 |
| mmproj 视觉状态不明 | 可能实际是 text only | 用图片请求和 `/status` 验证 |

### 9.2 建议的下一步

1. **先完成 Qwen3.8-27B 的干净基线**：8K/32K/64K/128K，Q4_K_M，INT8 KV，`verify-rows all`。
2. 补 Qwen3.6-35B-A3B 的 Splash 标准资产，再做同样上下文矩阵。
3. 向 Splash 维护方确认 `gemma4` 支持计划；在支持前不要把 Gemma 放进 Splash 对比表。
4. Flash-Next 只做维护窗口内的探索性实验，不作为正式交付指标。
5. 将每次运行的命令、模型 revision、文件哈希、Splash 版本、指标和失败原因写入结果文件。

### 9.3 机器独占通知建议

正式评测前通知同事：

```text
将在 192.168.3.13（SynCo-MacMini）进行 Splash 单模型性能评测，
期间需要独占 GPU/内存带宽，暂停其他大模型推理和重型任务。
预计每模型 2～4 小时；如遇长上下文或模型下载，时间会延长。
测试期间 PAIR 代理可以保留，但请不要发起大模型请求。
```

## 10. 最终结论

| 模型 | 是否可行 | 是否能发挥最大性能 | 建议 |
|---|---|---|---|
| Qwen3.8-27B | ✅ 现在可做 | ⚠️ Q4 下接近速度上限，非质量上限 | **立即作为首选基线** |
| Qwen3.6-35B-A3B | ⚠️ 补资产后可做 | ⚠️ 有 MoE 潜力，需独占与 DFlash2 | **第二优先，先补齐资产** |
| Gemma-4-26B-A4B | ❌ Splash 当前不可做 | ❌ 无法判断 | **等框架支持；不要用 MLX 结果冒充** |
| Qwen3.8-Flash-Next | ⚠️ 仅实验 | ❌ 不能 | **只做低比特/短上下文的探索性验证** |

**推荐启动顺序：Qwen3.8-27B → Qwen3.6-35B-A3B →（等待）Gemma-4-26B-A4B →（独立实验）Qwen3.8-Flash-Next。**

---

## 附录 A：核查依据

- Splash 1.5.0 安装目录：`/opt/homebrew/Cellar/splash/1.5.0/libexec/install/`
- 架构表：`gguf.py` 中 `ARCHITECTURES`
- 模型族映射：`families.py` 中 `FAMILIES`
- 启动示例：`launcher.py`
- 官方建议模型：`suggested-models.tsv`
- 3.13 模型目录：`/Users/synco/Models/splash/`
- 3.13 Ollama manifest：`/Users/synco/.ollama/models/manifests/registry.ollama.ai/library/qwen3.6/35b-a3b-q4_K_M`
- 3.7 Gemma GGUF：`/Users/admin/workspace/GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`
- 3.7 Gemma MLX：`/Users/admin/.omlx/models/gemma-4-26b-a4b-it-4bit`
- Qwen3.8-Flash-Next 官方 README 与 config
- ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO GGUF README
- Unsloth Qwen3.8-Flash-Next GGUF 仓库文件清单

## 附录 B：术语

| 术语 | 含义 |
|---|---|
| DFlash2 | Splash 使用的投机解码 draft 模型 |
| PLE | Per-Layer Embedding，Flash-Next 的 n-gram 表相关组件 |
| MoE | Mixture of Experts，按 token 只激活部分专家 |
| KV Cache | 注意力键值缓存，随上下文增长 |
| TTFT | Time To First Token |
| Prefill | 处理输入 prompt 的阶段 |
| Decode | 逐 token 生成输出的阶段 |
| mmproj | 多模态视觉投影器 |
| RSS | Resident Set Size，进程常驻内存 |
| Wired memory | 不可被系统换出的内存 |
| SSD streaming | 权重/表不常驻内存，按需从 SSD 读取 |

> 说明：本文中的 KV 估算基于公开 config 的层类型、头数和 head_dim，是理论上限/下限参考，不是 Splash 实测峰值。正式结果必须以实际进程 RSS、wired memory、swap 和 Splash 日志为准。
