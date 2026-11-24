# Mac mini M5 Pro 文本模型性能摸底评测方案（v5：仓库原生能力评测集 + 并发阶梯 + 机器指标实时监控）

> 项目仓库：`/Users/augustus/Desktop/LLMBenchmark`
> 评测目标：第一轮 Qwen3.8-27B（Splash 1.5.0）、Qwen3.6-35B-A3B（Splash 1.5.0）、Gemma-4-26B-A4B（MLX-VLM 0.8.0）；第二轮 Qwen3.8-Flash-Next（Splash 1.5.0，实验性探测）
> 评测维度：冷加载时间、TTFT / Prefill 速度、Decode 速度、内存占用（RSS / wired / swap）、SSD 流式读取、最大可用上下文、稳定性统计、机器性能影响分析
> 主测机：`192.168.3.5` / `Jiuwen-MacMini` / SSH alias `macmini-m5-jiuwen`（独占窗口）
> 核心原则：**不同推理引擎分表记录，不跨引擎直接排名**
> 日期：2026-11-24（v5）
> 本次修订依据：[Jiuwen3.5_Splash与Gemma_MLX评测部署方案.md](./Jiuwen3.5_Splash与Gemma_MLX评测部署方案.md)（2026-11-24，Splash/MLX-VLM 实机核查）
> 历史版本：v3 及更早版本（Ollama 统一基线 + MLX-LM 专项、三机对比口径）保留在 git 历史中，本版起不再维护该口径。
> v5 修订：评测集改用仓库原生能力评测集（GPQA、MMLU、MMLU-Pro、MATH500、ARC-C、GSM8K）；正式口径为相同 query 的并发阶梯式上涨（1→2→4→8→16），并在全程实时监控 CPU/GPU/NPU 使用率、内存、算力和带宽并输出时间序列图。以上述修订与后文 v5 标注内容为准。

---

## 〇、版本变更记录

### 〇.1 v5 相对 v4 的关键变更（本版）

| 变更项 | v4（2026-11-24） | v5（本版） |
|---|---|---|
| 评测集 | 官方全量数据构造固定输入 | **仓库原生能力评测集：GPQA / MMLU / MMLU-Pro / MATH500 / ARC-C / GSM8K** |
| 并发口径 | 单请求（`--queue-size 1`） | **并发阶梯 1→2→4→8→16，档位间分表分图，不混算** |
| 机器监控 | 关键点手动采样 | **全程实时采样 CPU/GPU/NPU、内存、算力、带宽，输出时间序列图（图 6～9）** |
| 汇总模板 | 单表汇总 | **按并发档位分表（表 A-c1…），并补"错误/超时"列** |

### 〇.2 v4 相对 v3 的关键变更

| 变更项 | v3（2026-11-23） | v4（2026-11-24） |
|---|---|---|
| 推理引擎 | Ollama 统一基线 + MLX-LM 性能专项 | **Splash 1.5.0（Qwen 系）+ MLX-VLM 0.8.0（Gemma）** |
| 主测机 | 3.5 / 3.13 / 3.7 三机 | **仅 3.5**；3.13 磁盘紧张、3.7 无 MLX-VLM 环境，退出本轮 |
| Qwen3.8-27B | 计划通过 Ollama/llama.cpp 测 | **Splash GGUF Q4_K_M + DFlash2，资产完整，首轮首选** |
| Qwen3.6-35B-A3B | 下载 MLX-LM 4bit 约 20.4GB | **复用已有 19GiB MLX 4bit，只补约 1.05GB DFlash2** |
| Gemma-4-26B-A4B | Ollama 部署 + MLX 备选 | **必须走 MLX-VLM**（Splash 不支持 `gemma4`） |
| Qwen3.8-Flash-Next | ❌ 不作本轮正式基线 | **第二轮实验性探测**（低比特/SSD 流式，缩减矩阵，表 D 单列不排名） |
| 新增指标 | — | 冷加载时间、wired memory、SSD 读写（流式检测）、error/timeout、finish_reason、均值/中位数/P95 |
| 对比口径 | 允许 Ollama 与 MLX 同表标注比较 | **强制分表**：同引擎内排名，跨引擎只做观察 |
| 资源前提 | 建议释放 3.13 Splash | **3.5 独占窗口是硬前提**：停止 Ollama 35B（约 30GiB）与 MLX 9B（约 5GiB） |

---

## 一、重要前置：机器与资产核查

### 1.1 主测机配置（3.5 实测）

| 项目 | 实测值 |
|---|---|
| 主机名 | `Jiuwen-MacMini` |
| 地址 / SSH alias | `192.168.3.5` / `macmini-m5-jiuwen` |
| Model Identifier | `Mac17,16` |
| 芯片 | Apple M5 Pro，18 CPU 核（6 Super + 12 Performance） |
| GPU | 20 核，Metal 4 |
| 统一内存 | 64GB |
| 系统版本 | macOS 27.2 |
| 数据盘 | 926GiB，可用约 454GiB（49% 使用率） |
| Splash | `/opt/homebrew/bin/splash`，1.5.0，无需重装 |
| MLX 运行时 | `/Users/jiuwen/mlx-bench-venv`：mlx 0.32.4 / mlx-vlm 0.8.0 / transformers 5.24.0 / Python 3.14.8 |

3.13（约 78GiB 可用磁盘）与 3.7（无 MLX-VLM 环境）本轮不作为主测机；跨机对比见第六节。

### 1.2 模型资产状态

| 模型 | 引擎 | 权重 | Draft/投影 | 状态 |
|---|---|---|---|---|
| Qwen3.8-27B | Splash | `/Users/jiuwen/Models/splash/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`（16,464,440,224 B ≈ 15.33GiB） | DFlash2 `/Users/jiuwen/Models/splash/Qwen3.8-27B-DFlash2/model.safetensors`（≈ 3.58GiB）+ mmproj BF16（≈ 887MiB） | ✅ 可立即 smoke |
| Qwen3.6-35B-A3B | Splash | `/Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit`（MLX 4bit affine，约 19GiB，已过 Splash 架构检查） | 缺 DFlash2：`incoai/Qwen3.6-35B-A3B-DFlash2` revision `51ef7b6923ad6c14cb1bb41c37a9041446496ab6`，约 1.05GB | ⚠️ 补 draft 后可用 |
| Gemma-4-26B-A4B | MLX-VLM | `/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit`（约 14GiB，3 分片完整） | — | ✅ 可立即 smoke |
| Qwen3.8-Flash-Next（第二轮） | Splash | 无完整权重资产 | — | ⚠️ 实验性：低比特（IQ2/Q2 类）或 SSD 流式，下载校验通过后才启动 |

注意：同目录其他 Q5/Q6/Q8 GGUF 多为 135～136 字节 LFS pointer，不是有效权重（先查文件大小）。

### 1.3 正式测试的环境底线

核查时 3.5 上有以下驻留服务，**正式测量前必须协调停止**：

| 服务 | 端口 | 资源占用 | 影响 |
|---|---:|---:|---|
| Ollama（已加载 `qwen3.6:35b-a3b-q4_K_M`，ctx 65536） | `127.0.0.1:11435` | llama-server RSS 约 30.2GiB | 挤压 KV 与 Metal 缓冲 |
| MLX-LM（已加载 `Qwen3.5-9B-4bit`） | `8090` | RSS 约 5.2GiB | 额外占用内存与 GPU |
| NVIDIA PAIR | `8080`、`11434` | 代理进程 | 可能继续触发 Ollama 请求 |

规则：

- **未取得机器使用方同意前，不得直接 kill 任何共享服务**；先 `ps`/`lsof` 记录，再协商；
- 正式口径：单模型、单实例；评测集统一使用仓库原生能力评测集（GPQA/MMLU/MMLU-Pro/MATH500/ARC-C/GSM8K）；并发按阶梯 `1→2→4→8→16` 上涨，`concurrency=1` 仅为低并发基线；
- 本轮性能端口约定：Splash Qwen3.8 → `18080`，Splash Qwen3.6 → `18081`，MLX-VLM Gemma → `18091`；
- 测完一个模型完整退出进程后再启动下一个；磁盘写入、下载、索引构建期间不测速。

### 1.4 结论

> 3.5 机器与资产足够。评测分两轮：**第一轮**依次测 Qwen3.8-27B（Splash，资产完整）→ Qwen3.6-35B-A3B（Splash，补 DFlash2）→ Gemma-4-26B-A4B（MLX-VLM）；**第二轮**为 Qwen3.8-Flash-Next 实验性探测（Q4 约 65~75GB 超出 64GB 内存，只做低比特/SSD 流式口径，不作正式基线）。**正式 benchmark 必须等独占窗口。**

---

## 二、评测维度定义

本次评测回答 8 个核心问题。

### 2.0 冷加载时间

**定义：** 进程启动到 `/ready`（或 `model_ready=true`）的耗时。

**为什么重要：** 决定权重加载、KV 预分配和引擎初始化的一次性成本；冷加载失败（EOF/OOM）是稳定性信号。

### 2.1 TTFT / Prefill 速度

**定义：** 输入 N 个 prompt tokens 后吐出第一个 token 的时间，或单位时间处理的输入 token 数。

```text
TTFT = first_token_time - request_start_time
Prefill Speed = prompt_tokens / prefill_duration   (tokens/sec)
```

### 2.2 Decode 速度

**定义：** 逐 token 生成吞吐。

```text
Decode Speed = completion_tokens / decode_duration   (tokens/sec)
```

**注意：** Splash 使用 DFlash2 投机解码，其 Decode 速度包含 draft 加速效应，**必须与不带 draft 的 MLX-VLM Decode 分开记录**，不得直接横比"谁更快"。

### 2.3 内存占用

**定义：** 推理期间进程峰值内存与系统压力。分四项记录：

1. 进程峰值 RSS（psutil 采样）；
2. wired memory（不可换出的锁定内存）；
3. swap 使用量与增量（`sysctl vm.swapusage`）；
4. MLX 系自报 peak memory（如有）。

### 2.4 SSD 流式读取（v4 新增）

**定义：** 权重/专家/DFlash2 是否在推理期间持续从 SSD 读取（内存不足信号）。

**测法：** 采样磁盘读写速率；Splash 场景重点观察 `--expert-memory` 不足导致的专家流式读取和 Flash-Next 的 PLE/n-gram 流式读取。

### 2.5 能力评测集与输入长度

**定义：** 使用仓库原生能力评测集作为实际 query，观察不同输入长度、难度和并发下的性能与机器资源。

**测法：** 按 `dataset_id` 统计实际 `prompt_tokens`，按长度分桶分析；长输入能力题单独探测，不默认进入正式主矩阵。模型配置的 `max_position_embeddings` 只是架构上限。

### 2.6 稳定性统计

**定义：** 至少 1 次 warm-up + 正式重复 3～5 次，报告均值、中位数、P95；记录 error/timeout 次数与 `finish_reason`（正常结束/长度截断/异常）。旧日志中的交互式请求不算正式基准。

### 2.7 机器性能影响分析

**定义：** 分析内存带宽、GPU 核数与架构、温控降频、SSD 流式读取、`--max-memory` 配置对速度的影响。见第七节。

### 2.8 机器资源实时监控（v5 新增）

在推理过程中同步采样并保存下述机器指标，最后输出时间序列图；每类指标都要区分“占用/总量”或给出单位：

| 指标 | 定义 / 采集方式 |
|---|---|
| GPU / NPU 使用率 | Apple 统一 GPU 与 ANE 的 duty cycle / 频率 / 功率，口径为占用/总量；有 sudo 时使用 `powermetrics --samplers gpu_power,ane_power`，无 sudo 显式标记“未采集” |
| CPU 使用率 | 各核心 user/sys/idle/nice 比例，记录占用核数/总核数；`top -l`、`ps`、可选 `powermetrics --samplers cpu_power` |
| 内存（含统一内存） | RAM 已用/空闲/缓存/可用、进程 RSS、swap；`vm_stat`、`sysctl hw.memsize`、`memory_pressure`、`sysctl vm.swapusage` |
| 算力（TFLOPS / PFLOPS） | 必须记录模型量化单位/`weight_format`；由参数规模与实测 token/时间换算有效浮点吞吐，理论峰值与实测分列，禁止把模型参数直接当作实测算力 |
| GPU / NPU 带宽 | 官方理论值（按规格记，如 120GB/s）与实测带宽；实测带宽须用内存带宽核验，不得用 tok/s 代替 |

这些指标按并发档位分面记录；“未采集”的字段标 `null`，不得用推测值补齐。

---

## 三、推理引擎选型（v4；v5 修订并发参数口径）

### 3.1 Splash 1.5.0（Qwen 系主引擎）

原生支持模型族：`Qwen3.8-27B`、`Qwen3.6-35B-A3B`、`Qwen3.5-122B-A10B`、`Qwen3.8-Flash-Next`、`Flint`；底层架构 `qwen35 / qwen35moe / qwen4exp / flint`。**不支持 `gemma4` / `gemma4_text`，无法通过改名绕过。**

`--model-dir` 同时支持本地 GGUF、本地 MLX affine 权重、DFlash2 draft 目录和 GGUF mmproj。

关键参数：

| 参数 | 作用 | 注意 |
|---|---|---|
| `--model` | API 显示 ID，必填 | 本地加载也必须提供 |
| `--model-dir` | 本地权重路径 | GGUF 文件或 MLX 目录 |
| `--draft-model` | DFlash2 draft 目录 | **不能跨模型复用**（Qwen3.8 的 draft 不能给 Qwen3.6 用） |
| `--language-only` | 文本模式 | 正式基线优先文本，避免视觉 tower 干扰 |
| `--max-memory` | 内存上限 | 40G/42G 是保守起点，按实测 RSS 调整；设过高会触发系统 swap |
| `--kv-format int8` | KV 精度 | 全程固定，不混测 FP16/INT8 |
| `--state-format bf16` | 状态精度 | 固定 |
| `--max-context` | 上下文上限 | smoke 用短能力题；正式按能力集与并发矩阵调整 |
| `--queue-size` | 并发队列 | 按并发档位设置（1→2→4→8→16）；冒烟与冷加载测量固定 1 |
| `--idle-release` | 空闲释放权重 | 共享机更安全，但复测重新加载（冷加载） |
| `--expert-memory` | MoE 专家 RAM 槽位（第二轮 Flash-Next 用） | 不足时专家从 SSD 流式读取，速度下降 |
| `--ple-memory` | Flash-Next PLE/n-gram 表缓存 | 不设或过小会流式读 SSD |
| `--sparse-kv-memory` | Flash-Next 稀疏 KV 缓存 | 影响长上下文性能 |

三个容易混淆的"支持"：**架构支持 ≠ 资产完整；能加载 ≠ 能稳定正式评测；能出 tok/s ≠ 发挥最大性能。**

### 3.2 MLX-VLM 0.8.0（Gemma 专用）

`/Users/jiuwen/mlx-bench-venv` 已含 `gemma4` / `gemma4_text` 实现。要点：

- `--max-num-seqs` 按并发档位设置（1→2→4→8→16），冒烟与冷加载测量固定 1，防并发污染；
- `--max-tokens` 是单次生成上限，不是上下文窗口；
- 就绪判断以 `/health` 返回 `model_ready=true` 为准。

### 3.3 Ollama 与 MLX-LM 的角色变化

- **Ollama：本轮不作为性能基线引擎。** 3.5 上已驻留的 Ollama 35B 服务（约 30GiB）属于必须清理的负载，其历史交互结果不作正式基准。
- **MLX-LM（mlx_lm.generate/server）：** 降级为交叉复核工具（如需验证 MLX affine 权重本身的 tok/s/peak memory）。与 MLX-VLM 是不同运行时，结果分开标注。

### 3.4 分表原则

| 表 | 引擎 | 模型 | 端口 |
|---|---|---|---|
| A | Splash 1.5.0 | Qwen3.8-27B（GGUF Q4_K_M + DFlash2） | 18080 |
| B | Splash 1.5.0 | Qwen3.6-35B-A3B（MLX 4bit + DFlash2） | 18081 |
| C | MLX-VLM 0.8.0 | Gemma-4-26B-A4B（MLX 4bit） | 18091 |

同引擎内可做模型排名；跨引擎只做观察（如"同输入长度下 TTFT 分布"），不做合并排名。

---

## 四、评测流程总览

```text
[Step 0] 独占窗口与负载清理
  → 与使用方确认独占窗口（不擅杀）
  → 记录 ps/lsof/memory_pressure/vm_stat/swap/df 基线
  → 停止 Ollama 35B 与 MLX-LM 9B；确认 18080/18081/18091 无冲突

[Step 1] Qwen3.8-27B Splash（表 A）
  → smoke test：/ready、/status、chat completion、日志无 EOF/OOM
  → 六类能力评测集 × 并发阶梯 1→2→4→8→16（warm-up 1 + 重复 3~5）
  → 长输入能力题单独探测

[Step 2] Qwen3.6-35B-A3B Splash（表 B）
  → 固定 revision 下载 DFlash2（约 1.05GB）并校验哈希
  → smoke test → 同一能力集与并发阶梯 → 长题探测

[Step 3] Gemma-4-26B-A4B MLX-VLM（表 C）
  → CLI smoke → 常驻服务 → /health model_ready
  → 同一能力集与并发阶梯 → 长题探测

[Step 4] 分引擎汇总
  → 表 A/B/C 各自汇总，不合并排名
  → 冷加载、TTFT、Prefill、Decode、内存、能力集稳定性、SSD 读取
  → 异常样本单独说明，不删除失败记录
```

第二轮（第一轮报告完成并经确认后，独立窗口）：

```text
[R2-Gate] 前置门槛
  → Flash-Next 低比特权重下载 + 哈希校验（未就绪则推迟）
  → 第一轮表 A/B/C 报告完成

[R2-1] Flash-Next Splash smoke（端口 18082）
  → 显式设置并记录 --expert-memory / --ple-memory / --sparse-kv-memory

[R2-2] 能力题缩减集并发探测 + 长题单独探测
  → 不默认执行超长输入（Q4 约 65~75GB 超出 64GB 内存）

[R2-3] 表 D 单独输出（实验性口径）
  → 重点记录 SSD 流式读取对 Decode 的影响与边界行为
  → 不与表 A/B/C 排名
```

---

## 五、详细评测脚本设计

### 5.1 环境核查（只读）

```bash
ssh macmini-m5-jiuwen

/opt/homebrew/bin/splash --version
/Users/jiuwen/mlx-bench-venv/bin/python -c "import mlx.core as mx, mlx_vlm; print(mx.__version__)"

ps -axo pid,rss,command | egrep 'ollama|mlx_lm|mlx_vlm|splash serve|llama-server' | egrep -v egrep
lsof -nP -iTCP -sTCP:LISTEN | egrep ':(8080|8090|11434|11435|18080|18081|18091)\b' || true
memory_pressure | tail -12
vm_stat | head -16
sysctl vm.swapusage
df -h /System/Volumes/Data
```

### 5.2 模型准备

**Qwen3.6 补 DFlash2（固定 revision）：**

```bash
mkdir -p /Users/jiuwen/Models/splash/Qwen3.6-35B-A3B-DFlash2

HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
  incoai/Qwen3.6-35B-A3B-DFlash2 \
  --revision 51ef7b6923ad6c14cb1bb41c37a9041446496ab6 \
  --include config.json \
  --include model.safetensors \
  --local-dir /Users/jiuwen/Models/splash/Qwen3.6-35B-A3B-DFlash2

ls -lh /Users/jiuwen/Models/splash/Qwen3.6-35B-A3B-DFlash2
shasum -a 256 /Users/jiuwen/Models/splash/Qwen3.6-35B-A3B-DFlash2/model.safetensors
```

校验点：`config.json` 含 `DFlash2DraftModel`，`model.safetensors` ≈ 1,052,513,456 bytes。**不要用 Qwen3.8 的 draft 代替。**

### 5.3 Splash 启动与就绪

**Qwen3.8-27B（端口 18080）：**

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

**Qwen3.6-35B-A3B（端口 18081）：**

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

**就绪判断（以 Splash 为例，端口换成对应值）：**

```bash
for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:18080/health >/dev/null 2>&1; then
    echo "Splash HTTP ready"; break
  fi
  sleep 1
done

curl -fsS http://127.0.0.1:18080/ready
curl -fsS http://127.0.0.1:18080/status
```

就绪以 `/ready` 成功、`/status` 中模型与上下文正确为准，不能只看端口监听。

**冒烟请求：**

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

冒烟成功标准：HTTP 2xx、回答正常、`/status` 正确、RSS/swap 可解释、日志无 `Engine failed`/EOF/OOM/重复重启。历史 `run.log` 中的 EOF 只能作参考，正式前必须重新 smoke 并保存新日志。

### 5.4 MLX-VLM 启动与就绪（Gemma，端口 18091）

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

# 就绪判断：/health 返回 model_ready=true
for i in $(seq 1 180); do
  health="$(curl -fsS http://127.0.0.1:18091/health 2>/dev/null || true)"
  if printf '%s' "$health" | python3 -c \
    'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("model_ready") is True else 1)' \
    2>/dev/null; then
    echo "Gemma-4 ready"; break
  fi
  sleep 1
done
```

### 5.5 Prefill / Decode 测量

两个引擎适配器都走 OpenAI 兼容 `/v1/chat/completions`：

- **非流式**：从响应 usage/计时字段取 prompt_tokens、completion_tokens 与服务端耗时，算 Prefill/Decode tok/s；
- **流式（`stream: true`）**：客户端记录 `request_start`、首 chunk、末 chunk 时间戳，计算 TTFT 与 E2E；两套都要跑，口径分开记录；
- 字段缺失时记 `null`，**不补 0、不估算**；不要假定两个引擎返回相同的 timings 字段。

### 5.6 内存与 SSD 测量

```bash
# 进程峰值 RSS（psutil 采样，0.5s 间隔）
python3 - <<'RSS'
import psutil, time
proc = psutil.Process(<PID>)
peak = 0
while proc.is_running():
    peak = max(peak, proc.memory_info().rss)
    time.sleep(0.5)
print(f"Peak RSS: {peak / 1024**3:.2f} GiB")
RSS

# 系统内存压力 / swap
memory_pressure
vm_stat            # wired 内存、free/active/inactive/cache
sysctl vm.swapusage
sysctl hw.memsize  # 统一内存总量，用于“占用/总量”口径

# SSD 读写采样（判断是否流式读取权重/专家）
iostat -w 1 -d disk0 60

# CPU/GPU/NPU 实时采样（需要 sudo；无权限时标记“未采集”）
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 60 \
  -o results/mac_benchmark/monitor/powermetrics.txt
```

每档能力评测与并发同时记录：峰值 RSS、wired memory、swap 增量、SSD 读写速率、CPU/GPU/NPU、算力、带宽、error/timeout、finish_reason。

### 5.7 能力评测集与固定参数

正式 query 使用仓库原生能力评测集（见 5.9），不再按固定上下文阶梯构造输入。每个并发档位均记录：cold load time、TTFT、Prefill tok/s、Decode tok/s、`prompt_tokens`、`completion_tokens`、峰值 RSS、wired、swap、SSD 读写、CPU/GPU/NPU、算力、带宽、error、`finish_reason` 与判分结果。

固定参数：

```text
temperature = 0
top_p = 1
max_tokens = 固定值（各档一致）
stream = 固定
KV format = int8（Splash 全程固定）
context limit = 冒烟 8K；正式统一 32K（与 8.1 JSON 模板 context_tokens 一致），内存受限时按实测下调并记录
queue/concurrency = 1 → 2 → 4 → 8 → 16（阶梯上涨；1 为基线）
warm-up ≥ 1 次
repeat = 3~5 次
```

### 5.8 脚本结构

```text
tools/
  data_collector/       # 仓库原生能力评测集采集与判分入口
  bench_common.py       # 客户端计时、统计（均值/中位数/P95）、JSON schema
  bench_splash.py       # Splash 18080/18081 适配器
  bench_mlxvlm.py       # MLX-VLM 18091 适配器
  bench_multi_query.py  # 能力集并发阶梯多 query
  monitor_machine.py    # CPU/GPU/NPU/内存/算力/带宽实时监控
  summarize.py          # 分引擎汇总 Markdown 与图表
```

### 5.9 仓库原生能力评测集与并发多 Query 口径（v5 新增）

正式 query 使用仓库原生能力评测集，不再使用 `benchmark_data/performance/` 性能 workload：

| dataset_id | 数据文件 | 全量条数 | 判分方式 |
|---|---|---:|---|
| `gpqa` | `data/GPQA/gpqa_diamond.json` | 198 | 选项匹配 |
| `mmlupro` | `data/MMLUPro/test.json` | 1001 | 选项匹配 |
| `math500` | `data/MATH500/test.json` | 500 | 答案抽取 + 数学判分 |
| `arcc` | `data/arc_c/arc_test.json` | 1172 | 选项匹配 |
| `gsm8k` | `data/GSM8K/test.json` | 100 | 数值匹配 |
| `mmlu` | `data/MMLU/test.json` | 182 | 选项匹配 |

以上文件是 JSONL（扩展名仍为 `.json`）。配置入口为 `config/edge_router_config.yaml` 的 `datasets` 段，`dataset_id` + `splits: [test]`。三个模型共用同一批能力 query，按 `(dataset_id, record_index)` 严格对齐；本轮既记录性能与机器资源，也保留仓库 evaluator 的判分结果。

并发阶梯：

```text
concurrency = 1 → 2 → 4 → 8 → 16
```

每档先预热，再重复下发同一批能力 query；客户端记录 per-request TTFT/E2E/completion tokens/`finish_reason`，并汇总该档请求吞吐、成功 token 吞吐、排队时间和错误率。Splash `--queue-size` 与 MLX-VLM `--max-num-seqs` 按当前档位设置；若某档出现 OOM、严重排队、swap 暴涨或雪崩，停止升档并保留上一档为该机器/模型下的稳定最大并发。**不同并发档位的数据分表、分图，不得混算。**

本地模型配置与入口：

```bash
# 从 config/local_model_eval.yaml 启动
/Users/augustus/miniconda3/bin/python tools/run_local_bench.py   --model-name qwen3.8-27b --limit-per-dataset 0

# 汇总并成图
/Users/augustus/miniconda3/bin/python tools/summarize_multi_query.py   --result-dir results/mac_benchmark/multi_query/qwen3.8-27b   --chart
```

运行方式：

```bash
python tools/monitor_machine.py --duration 3600 --interval 2   --out results/mac_benchmark/monitor/<model_tag> --chart

python tools/bench_multi_query.py   --model 'mlx-community/Qwen3.6-35B-A3B-4bit'   --port 18081   --limit-per-dataset 0   --concurrency 1,2,4,8,16   --out results/mac_benchmark/multi_query/<model_tag>
```

---

## 六、跨引擎对比原则

### 6.1 可以横向比较

- 同一引擎、同一机器、同一量化精度下的模型（表 A vs 表 B）；
- 同一输入/输出长度下的 TTFT / Prefill / Decode；
- 同一能力 query 长度桶与并发档位下的峰值内存与稳定性；
- 同一 warm-up 与重复次数后的统计分布。

### 6.2 不应横向比较

- Splash（DFlash2 投机解码）与 MLX-VLM（普通 decode）的 tok/s 直接排名；
- GGUF Q4_K_M 与 MLX 4bit affine 宣称"模型能力差异"；
- 不同 KV 精度、不同上下文、不同 reasoning effort 混表；
- 冷启动单次与 warm 后多次混在一起；
- 并发与单请求混在一起；
- 只写"某模型 tok/s"而不写引擎。

### 6.3 跨机对比的处理

本轮主测机为 3.5，单机单模型独占。M4 Pro（3.7）与 3.13 的跨机对比**延后到本轮完成后**作为可选阶段，且必须满足：相同引擎、相同模型/量化、相同上下文与输出长度；3.13 磁盘紧张、Splash/磁盘风险高，不与 3.5 直接混排。不同时改变"芯片 + 引擎 + 量化"三个变量。

---

## 七、性能归因分析

### 7.1 关键硬件与运行时因素

| 因素 | 对 Prefill 的影响 | 对 Decode 的影响 | 本轮如何测量 |
|---|---|---|---|
| 内存带宽 | 高 | **很高，通常 Decode 主瓶颈** | 同模型同引擎同能力 query 长度桶对比 tok/s |
| GPU 核数与架构 | **很高（主瓶颈）** | 中高 | M5 Pro 单机内不同模型/能力 query 对比 |
| 统一内存容量 | 中 | 高（决定 KV 上限） | 能力集长题 OOM 点 |
| KV 精度与实现 | 中 | 高 | `--kv-format int8` 全程固定 |
| DFlash2 投机解码 | 低 | **高（Splash 特有）** | 记录 draft 对 Decode 的影响，表 A/B 单独标注 |
| expert-memory 槽位 | 中 | 高 | Splash `--expert-memory`；不足时专家走 SSD |
| SSD 流式读取 | 冷启动明显 | swap/流式时很高 | `iostat` 采样 |
| 温控降频 | 中 | 中高 | 连续 Decode 多轮比较首轮与末轮 |
| `--max-memory` 设置 | 中 | 高 | 过高触发系统 swap；从 40G/42G 起步调整 |
| ANE | 低 | 低 | Splash/MLX 主要走 GPU/Metal |

### 7.2 归因测试方法

```bash
sysctl -n hw.model hw.memsize
vm_stat
sysctl vm.swapusage
memory_pressure
iostat -w 1 -d disk0
sudo powermetrics --samplers cpu_power,gpu_power,thermal -i 1000   # 需 sudo；无权限标"未采集"
```

### 7.3 结论模板

正式报告应回答：

1. **Decode 瓶颈**：各模型 Decode tok/s 与内存带宽的关系；DFlash2 带来多少加速；
2. **Prefill 瓶颈**：不同输入长度下 TTFT 与 Prefill tok/s 如何变化；
3. **内存瓶颈**：权重、DFlash2、KV Cache、Metal 缓冲各占多少；是否 swap；
4. **能力集稳定性**：每个 dataset 与并发档位是否稳定通过；失败发生在加载、Prefill 还是 Decode；长题探测结果单独记录；
5. **SSD 流式**：是否出现专家/权重流式读取，与 `--expert-memory`/`--max-memory` 的关系；
6. **稳定性**：重复测试均值、中位数、P95 和最差一次，不只报最好成绩。

---

## 八、输出格式

### 8.1 单次运行 JSON 结果

```json
{
  "engine": "splash",
  "engine_version": "1.5.0",
  "model": "Qwen/Qwen3.8-27B",
  "model_revision_or_digest": "UD-Q4_K_M / 文件 sha256",
  "weight_format": "gguf",
  "quantization": "Q4_K_M",
  "draft_model": "Qwen3.8-27B-DFlash2",
  "kv_format": "int8",
  "state_format": "bf16",
  "hardware": {
    "host": "Jiuwen-MacMini",
    "chip": "Apple M5 Pro",
    "model_identifier": "Mac17,16",
    "memory_gib": 64,
    "os_version": "27.2"
  },
  "context_tokens": 32768,
  "max_output_tokens": 128,
  "temperature": 0,
  "seed": 42,
  "cold_load_s": null,
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
    "wired_gib": null,
    "swap_delta_gib": null,
    "memory_pressure": null
  },
  "ssd_read_mib_s": null,
  "error_count": 0,
  "finish_reason": null,
  "success": true,
  "notes": ""
}
```

所有 `null` 正式运行后替换为实测值；缺失字段不补 0、不估算。MLX-VLM 的行用 `engine: "mlx-vlm"`、`draft_model: null`、`kv_format: null`。

并发多 Query 能力评测结果（`tools/bench_multi_query.py` 的 `stage_<c>_requests.jsonl`）新增以下端侧评测字段：

```text
sample_id / eval_name / model_name / prompt / response
correct / predicted_answer / gold_answer / fail_reason
prompt_tokens / completion_tokens / reasoning_tokens
prefill_tokens / decode_tokens / prefill_tps / decode_tps
request_start_time / first_token_time / response_end_time
server_wall_s / client_e2e_s
cache_status / cache_hit_tokens / error / retry_count

run_id / host / engine / engine_version
weight_format / quantization / draft_model
kv_format / state_format / model_revision_or_digest
attempt_index / backoff_s / timeout_setting_s
error_type / http_status
per_request_machine.memory_used_gib / memory_free_gib / wired_gib / swap_used_mib
```

其中 `request_*` 使用 ISO 时间戳；耗时字段另用客户端 `time.monotonic()` 计算并写入 `client_e2e_s`，不能混用壁钟。

### 8.2 分引擎汇总表（v5 强制）

每个并发档位（1/2/4/8/16）复制一份本表，命名为表 A-c1、表 A-c2 …… 表 A-c16；**不同档位数据分表，不混算**。CPU/GPU/NPU、算力、带宽等机器指标随档位由 `monitor_machine.py` 输出时间序列图（见 8.3 图 6～9），不在本表重复列。

**表 A — Splash 1.5.0 / Qwen3.8-27B（GGUF Q4_K_M + DFlash2 + INT8 KV，单档模板）：**

| 能力评测集 | 条数 | cold load | TTFT | Prefill tok/s | Decode tok/s | 峰值 RSS | wired | swap | SSD 读 | 错误/超时 | 成功 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GPQA | 198 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MMLU | 182 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MMLU-Pro | 1001 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MATH500 | 500 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ARC-C | 1172 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GSM8K | 100 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

表 B（Splash / Qwen3.6-35B-A3B，MLX 4bit + DFlash2）与表 C（MLX-VLM 0.8.0 / Gemma-4-26B-A4B，MLX 4bit）用同一模板。**三表不合并排名；跨引擎只做观察性对比并显式标注引擎差异。**

**表 D（第二轮实验性）**：Splash / Qwen3.8-Flash-Next 低比特，使用能力评测集短-中长度题做并发探测，较长题单独探测，单列输出，不与第一轮排名；启动参数 `--expert-memory` / `--ple-memory` / `--sparse-kv-memory` 必须逐次记录。

### 8.3 图表输出

- 图 1：TTFT 与 Prefill tok/s 随输入长度变化（分引擎曲线）；
- 图 2：Decode tok/s 随输出长度变化（分引擎曲线）；
- 图 3：峰值内存（RSS + wired）与 KV 增量随能力 query 长度桶变化，标记 OOM/长题探测点；
- 图 4：同引擎内模型对比（表 A vs 表 B）；
- 图 5：SSD 读取速率与上下文/`--max-memory` 的关系；
- 图 6：CPU 使用率随时间与并发档位变化（占用/总核）；
- 图 7：GPU/NPU duty/功率随时间与并发档位变化；
- 图 8：内存 used/free/cache/available 与 swap 随时间变化；
- 图 9：吞吐与算力（TFLOPS/PFLOPS）随并发档位变化，标注量化单位。

图表保存到：`/Users/augustus/Desktop/LLMBenchmark/results/mac_benchmark/charts/`

---

## 九、与 LLMRouterBench 主项目的衔接

1. **模型能力层**：能力采集继续走各自引擎的 OpenAI 兼容接口；同一模型不同量化分别记录，不混成一条模型记录。
2. **性能成本层**：冷加载、TTFT、Prefill/Decode tok/s、峰值内存、可用上下文作为路由的延迟/内存/上下文约束；本地部署无 API 计费，成本口径与云端 `pricing` 分开。
3. **硬件节点层**：本轮所有结果绑定 `host=Jiuwen-MacMini`；后续扩到 3.13/3.7 时同一模型在不同节点是不同执行配置。

每次实验保存三层 fingerprint：

```json
{
  "hardware_fingerprint": {
    "host": "Jiuwen-MacMini",
    "chip": "Apple M5 Pro",
    "model_identifier": "Mac17,16",
    "memory_gib": 64,
    "os_version": "27.2"
  },
  "engine_fingerprint": {
    "name": "splash",
    "version": "1.5.0",
    "port": 18080
  },
  "model_fingerprint": {
    "name": "Qwen/Qwen3.8-27B",
    "format": "gguf",
    "quantization": "Q4_K_M",
    "draft_model": "Qwen3.8-27B-DFlash2",
    "kv_format": "int8",
    "digest": "..."
  }
}
```

执行配置命名区分"模型身份"与"执行节点/引擎/量化/上下文"，例如：

```text
Qwen3.8-27B-Splash-Q4KM-DFlash2-INT8KV-GPQA-Jiuwen35
Gemma-4-26B-A4B-MLXVLM-4bit-MMLU-Jiuwen35
```

---

## 十、执行清单

### P0：正式测试前必须完成

- [ ] 与 3.5 使用方确认独占窗口；
- [ ] 协商停止 NVIDIA PAIR 模型负载、Ollama Qwen3.6（约 30GiB）、MLX-LM 9B（约 5GiB）；
- [ ] 确认 `18080`、`18081`、`18091` 无冲突；
- [ ] 记录 `vm_stat`、`memory_pressure`、swap、磁盘空间基线；
- [ ] Qwen3.8 Splash smoke test：`/ready`、`/status`、chat completion、日志正常；
- [ ] 固定模型 revision、量化、KV、能力评测集、生成参数并写入配置。

### P1a：Splash 模型（表 A/B）

- [ ] Qwen3.8-27B：六类能力评测集 + 并发阶梯；
- [ ] Qwen3.8-27B：长输入能力题单独探测；
- [ ] 下载并校验 Qwen3.6 DFlash2（固定 revision）；
- [ ] Qwen3.6-35B-A3B：同一能力集与并发阶梯 + 长题探测；
- [ ] 记录 DFlash2 对 Decode 的影响；
- [ ] 记录 expert/权重是否发生 SSD 流式读取。

### P1b：Gemma MLX（表 C）

- [ ] Gemma-4-26B-A4B CLI smoke test；
- [ ] `--max-num-seqs` 按并发档位设置（冒烟与冷加载固定 1）；
- [ ] 六类能力评测集 + 并发阶梯 + 长题探测；
- [ ] 记录 peak memory、RSS、wired、swap；
- [ ] 标注 `engine=MLX-VLM 0.8.0`。

### P2：汇总

- [ ] 按引擎分表（A/B/C）；
- [ ] 同引擎内比较模型；跨引擎只做观察；
- [ ] 报告冷加载、TTFT、Prefill、Decode、内存、能力集稳定性、SSD 读取；
- [ ] 输出每个并发档位的 CPU/GPU/NPU、内存、算力、带宽原始采样与图表；
- [ ] 保存原始日志、命令、模型 revision 和测试日期；
- [ ] 异常样本单独说明，不删除失败记录。

### P3：第二轮 Qwen3.8-Flash-Next（第一轮完成并经确认后启动）

- [ ] 低比特权重下载、文件大小与哈希校验（未就绪则推迟）；
- [ ] Splash 启动并逐次记录 `--expert-memory` / `--ple-memory` / `--sparse-kv-memory`；
- [ ] 能力题缩减集并发探测 + 长题单独探测；
- [ ] 表 D 单独输出，标注实验性口径，不参与任何排名。

---

## 十一、Gemma-4-26B-A4B MLX-VLM 专项说明

### 11.1 为什么必须走 MLX

Gemma 配置为 `Gemma4ForConditionalGeneration` / `gemma4`（text：`gemma4_text`），Splash 1.5.0 明确不支持该架构，也不能通过修改模型名或换目录绕过。因此按既定规划使用 MLX-VLM 0.8.0（`/Users/jiuwen/mlx-bench-venv`，已含 `gemma4` 实现）。

### 11.2 已知冒烟基线

此前完成的 MLX-VLM 冒烟验证：

| 指标 | Gemma-4-26B-A4B |
|---|---:|
| `/health` | `model_ready=true` |
| `/v1/models` | 通过 |
| Chat Completions | 通过 |
| 峰值内存 | 约 15.42GB |

该数据只证明加载链路与接口正常，**不是正式性能基线**。

### 11.3 测试注意

- `--max-num-seqs` 按并发档位设置（冒烟与冷加载固定 1，防并发污染）、`temperature=0`；
- 结果标签写清 `MLX-VLM 0.8.0 + MLX 4bit affine`，不要只写"Gemma 性能"；
- 不能把 MLX 结果填进 Splash 表，也不能用 Gemma 的 tok/s 与 Qwen 的 Splash tok/s 直接排名。

---

## 十二、风险红线与历史版本说明

### 12.1 风险清单

| 风险 | 级别 | 说明 | 处理 |
|---|---|---|---|
| 共享服务未释放 | 高 | 结果不可复现，内存/带宽受污染 | 协调独占窗口，不擅杀 |
| Qwen3.8 历史 Splash EOF | 中高 | 旧日志有引擎重启记录 | 重新 smoke，保留新日志 |
| Qwen3.6 缺 DFlash2 | 中 | 无法走 Splash 完整 draft 路径 | 下载固定 revision 约 1.05GB draft |
| 权重格式混用 | 中 | GGUF Q4_K_M 与 MLX 4bit affine 不可混表 | 标注 `weight_format`，必要时补对照 |
| `--max-memory` 过高 | 中 | 触发系统 swap，速度失真 | 从 40G/42G 起步按实测调整 |
| 下载/写入期间测速 | 中 | SSD I/O 污染计时 | 错峰；测速窗口内无下载 |
| LFS pointer 误用 | 低 | 135 字节文件不是权重 | 先 `ls -lh` 验证大小 |
| Flash-Next 资产未就绪 | 中高 | 第二轮硬门槛 | 下载校验完成前不启动 |
| 低比特结果被误读为模型质量 | 中 | 表 D 口径 | 明确标注 IQ2/Q2 实验性，不代表模型真实能力 |

### 12.2 历史版本说明

- **v3（2026-11-23）及更早**：Ollama 统一基线 + MLX-LM 性能专项 + 三机（3.5/3.13/3.7）对比口径，原文见 git 历史（commit `0e76b3d`）；其中 Ollama 部署细节、Gemma Ollama 选型、3.7 oMLX 缓存分析等内容随引擎口径切换不再维护。
- **v4（2026-11-24）**：切换为 Splash 1.5.0 + MLX-VLM 0.8.0 双引擎、3.5 单机独占、分表对比口径。部署细节的完整实机核查证据见 [Jiuwen3.5_Splash与Gemma_MLX评测部署方案.md](./Jiuwen3.5_Splash与Gemma_MLX评测部署方案.md)。
- **v5（2026-11-24，本版）**：评测集切换为仓库原生能力评测集，正式口径改为并发阶梯（1→2→4→8→16），新增机器指标（CPU/GPU/NPU、内存、算力、带宽）全程实时监控；8.2 汇总表按并发档位分表。
