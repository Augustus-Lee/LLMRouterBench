# Mac mini M5 Pro 文本模型性能评测方案

> 目标机器：`192.168.3.5` / `Jiuwen-MacMini` / Apple M5 Pro / 64GB 统一内存  
> 评测范围：第一轮为三个本地量化文本模型的单模型推理性能；第二轮为 Qwen3.8-Flash-Next 实验性探测（125B-A6B，低比特 / SSD 流式）  
> 推理引擎：Splash 1.5.0（Qwen 系）+ MLX-VLM 0.8.0（Gemma）  
> 编制日期：2026-11-24  
> 本次修订：评测集改为仓库原生能力评测集（GPQA、MMLU、MMLU-Pro、MATH500、ARC-C、GSM8K）；正式口径为相同 query 的并发阶梯式上涨（1→2→4→8→16），并在全过程实时监控 CPU/GPU/NPU 使用率、内存、算力与带宽，输出机器指标时间序列图。

## 1. 结论摘要

**总体结论：方案有条件可行，可以进入正式测试。**

硬前置条件是取得 `192.168.3.5` 的独占窗口，停止或隔离机器上现有的 Ollama、MLX-LM、NVIDIA PAIR 等推理负载，并保持“单模型、单实例”。正式测量改为使用仓库原生能力评测集（GPQA、MMLU、MMLU-Pro、MATH500、ARC-C、GSM8K），按并发阶梯 `1→2→4→8→16` 同时下发多个 query，观察吞吐、排队时延、TTFT/Decode 和机器资源曲线的饱和点；单请求仅作为最低并发基线。否则内存、GPU/Metal 和内存带宽会被共享负载污染，测得的 tok/s 和峰值内存不具备可比性。

| 模型 | 引擎 | 当前资产 | 结论 |
|---|---|---|---|
| Qwen3.8-27B | Splash 1.5.0 | GGUF Q4_K_M、mmproj、DFlash2 完整 | 可行，首轮基线 |
| Qwen3.6-35B-A3B | Splash 1.5.0 | MLX 4bit affine 主模型完整；DFlash2 已下载并校验 | 可行，资产已完整 |
| Gemma-4-26B-A4B | MLX-VLM 0.8.0 | MLX 4bit affine，约 14GB | 可行，必须使用 MLX-VLM |
| Qwen3.8-Flash-Next（第二轮） | Splash 1.5.0 | 当前无完整权重资产 | ⚠️ 实验性探测：仅低比特（IQ2/Q2 类）或 SSD 流式口径 |

Gemma 的架构为 `gemma4` / `gemma4_text`，Splash 1.5.0 不支持，不能通过改模型名或目录绕过，因此采用双引擎方案。

**评测分两轮执行：** 第一轮只测上表前三个模型（资产与引擎均已就绪）；Qwen3.8-Flash-Next 的 Q4 权重约 65～75GB，超出 64GB 单机可容纳范围，放入第二轮做低比特 / SSD 流式实验性探测——待第一轮报告完成且权重资产下载校验通过后，再启动第二轮。


## 2. 评测目标与边界

### 2.1 本轮目标

测量每个模型的冷加载时间、TTFT、Prefill/Decode 速度、峰值内存、wired memory、swap、SSD 流式读取、稳定上下文上限、稳定性和硬件性能归因。

第一轮覆盖 Qwen3.8-27B、Qwen3.6-35B-A3B、Gemma-4-26B-A4B。第二轮 Qwen3.8-Flash-Next 的目标收敛为三点：SSD 流式读取对 Decode 的影响、低比特量化的 Decode 能力、长题边界行为；结果单列实验性口径。

本轮结果属于 LLMRouterBench 的“性能成本层”，可作为路由器的延迟、内存和上下文约束，不是模型质量榜单。


## 3. 机器与环境

| 项目 | 实测值 |
|---|---|
| 主机名 / 地址 | `Jiuwen-MacMini` / `192.168.3.5`（SSH alias：`macmini-m5-jiuwen`） |
| 芯片 / CPU / GPU | Apple M5 Pro / 18 CPU 核 / 20 GPU 核，Metal 4 |
| 统一内存 / 系统 | 64GB / macOS 27.2 |
| 数据盘 | 926GiB，核查时可用约 454GiB |
| Splash | `/opt/homebrew/bin/splash`，1.5.0 |
| MLX 环境 | `/Users/jiuwen/mlx-bench-venv`，mlx 0.32.4 / mlx-vlm 0.8.0 |

最近一次核查发现以下共享负载：

| 服务 | 资源 | 影响 |
|---|---:|---|
| Ollama Qwen3.6-35B-A3B | llama-server RSS 约 30～33GiB | 挤占统一内存、Metal 缓冲和内存带宽 |
| MLX-LM Qwen3.5-9B | RSS 约 5GiB | 额外占用 GPU 和内存 |
| NVIDIA PAIR | 代理进程 | 可能继续触发 Ollama 请求 |

核查时系统内存接近满载，并已出现 swap。**未取得使用方同意前不得直接停止共享服务。**正式窗口开始前先记录进程、端口、内存和磁盘基线，协商停止或隔离负载后再启动待测模型；测试期间禁止下载、复制大文件、编译或其他推理任务。

## 4. 评测指标

| 指标 | 定义 / 采集方式 |
|---|---|
| cold load | 进程启动到 `/ready` 或 `model_ready=true` 的耗时 |
| TTFT | 请求开始到首个输出 token，使用流式请求时间戳 |
| Prefill tok/s | 输入 token 处理速度；优先使用引擎 timing |
| Decode tok/s | 输出 token 生成速度；优先使用引擎 timing，否则用流式 chunk 计算并标注 `client_stream` |
| prompt/completion tokens | 响应 usage 中的实际 token 数 |
| peak RSS | 推理进程峰值常驻内存，psutil 每 0.5s 采样 |
| wired memory | 系统锁定内存，使用 `vm_stat` 等采样 |
| swap delta | 测试前后 swap 增量 |
| SSD I/O | 读写速率和总量，使用 `iostat`，判断权重或专家是否流式读取 |
| 稳定性 | error、timeout、`finish_reason`、重复重启和日志异常 |
| GPU / NPU 使用率 | Apple 统一 GPU 与 ANE 的 duty cycle / 频率 / 功率，口径为“占用/总量”；有 sudo 时 `powermetrics --samplers gpu_power,ane_power`，无 sudo 显式标记“未采集” |
| CPU 使用率 | 各核心 user/sys/idle/nice 比例，记录“占用核数/总核数”；`top -l`、`ps`，可选 `powermetrics --samplers cpu_power` |
| 内存（含统一内存） | RAM 已用/空闲/缓存/可用，进程 RSS，swap；`vm_stat`、`sysctl hw.memsize`、`memory_pressure`、`sysctl vm.swapusage` |
| 算力（TFLOPS / PFLOPS） | 必须记录模型量化单位/`weight_format`；由参数规模与实测 token/时间换算有效浮点吞吐，理论峰值与实测分列，禁止直接把模型参数当作实测算力 |
| GPU / NPU 带宽 | 官方理论值（按规格，如 120GB/s）与实测带宽；实测带宽须用内存带宽核验，不得用 tok/s 代替 |

每个场景预热至少 1 次，正式重复 3～5 次，报告均值、中位数、P95 和最差值。失败样本必须保留；引擎未返回的字段记为 `null`，不得补 0 或估算。

## 5. 测试矩阵与固定参数

### 5.1 能力评测集与执行矩阵

正式 query 使用第 6 节六类仓库原生能力评测集，不再按固定上下文阶梯构造输入。每个数据集按实际 query 长度统计服务端 `prompt_tokens` / `completion_tokens`，并按长度分桶分析（如 `<=1K` / `1K-4K` / `4K-16K` / `>16K`），不做 256K 性能探测。

模型配置的 `max_position_embeddings` 只是架构上限；能力集的实际输入长度以服务端 `usage` 为准。长上下文能力题如 MATH500/MMLU-Pro 中较长输入，单独记录成功、OOM、swap 与 `finish_reason`。

**第二轮（Qwen3.8-Flash-Next）**：使用 GPQA、MMLU 等短-中长度能力题做并发阶梯探测；较长能力题单独探测，不默认进入正式主矩阵。

### 5.2 固定参数

```text
temperature = 0
top_p = 1
seed = 42
max_tokens = 256（主矩阵；关键点补 512 复核）
stream = 固定，流式与非流式分别记录
reasoning effort = 固定
Splash KV format = int8；state format = bf16
queue/concurrency = 1 → 2 → 4 → 8 → 16（阶梯上涨；1 仅为基线）
MLX-VLM --max-num-seqs = 当前阶梯并发数（单请求基线为 1）
warm-up >= 1；repeat = 3~5
```

主矩阵统一加载第 6 节的仓库原生能力评测集，以服务端实际 `prompt_tokens` 为准；不传评测集时保留原合成 prompt，仅用于兼容和冒烟，不进入正式结果。

第二轮 Splash 启动额外固定并逐次记录：`--expert-memory`（MoE 专家 RAM 槽位）、`--ple-memory`（PLE/n-gram 表缓存）、`--sparse-kv-memory`（稀疏 KV 缓存）。三项设置不足时权重或表会从 SSD 流式读取——量化这种流式读取对 Decode 的影响正是第二轮目的之一，因此取值必须写入 engine_fingerprint。

### 5.3 并发阶梯与实时监控（本次修订新增）

正式性能口径采用同一批能力评测 query，**并发阶梯式上涨**：

```text
concurrency = 1 → 2 → 4 → 8 → 16
```

每个档位：

1. 预热 1 次后正式重复；
2. 并发工作线程从能力评测集按批次取 query，持续下发同一批 prompt；
3. 客户端记录 per-request TTFT、E2E、completion tokens、`finish_reason`，并汇总该档请求吞吐和成功 token 吞吐；
4. 服务端队列/解码并发跟随档位：Splash `--queue-size` ≥ 当前并发，MLX-VLM `--max-num-seqs` = 当前并发；都写入 fingerprint；
5. 升档规则：当前档无 OOM、无异常 swap，错误率可解释；若出现雪崩、严重排队或 swap 暴涨，则停止升档，保留上一档作为该机器/模型下的稳定最大并发。

与并发测量同时运行系统指标采集（看到 8.3），输出按并发档位分面的时间序列图；**不同并发档位的数据不得混表或合并统计**。单请求数据仍作为最低并发基线独立成表。


## 6. 能力评测集：仓库原生数据集

### 6.1 本轮采用的能力集

本轮不再使用 `benchmark_data/performance/` 下构造的性能 workload，改用仓库原生支持的六类能力评测集：

| dataset_id | 数据文件 | 全量条数 | 判分方式 |
|---|---|---:|---|
| `gpqa` | `data/GPQA/gpqa_diamond.json` | 198 | 选项匹配 |
| `mmlupro` | `data/MMLUPro/test.json` | 1001 | 选项匹配 |
| `math500` | `data/MATH500/test.json` | 500 | 答案抽取 + 数学判分 |
| `arcc` | `data/arc_c/arc_test.json` | 1172 | 选项匹配 |
| `gsm8k` | `data/GSM8K/test.json` | 100 | 数值匹配 |
| `mmlu` | `data/MMLU/test.json` | 182 | 选项匹配 |

说明：

- 这些文件是 JSONL，扩展名仍保留为 `.json`；
- 配置入口为 `config/edge_router_config.yaml` 的 `datasets` 段，`dataset_id` + `splits: [test]`；
- 本轮以这些能力题作为实际 query 输入，既测量性能与机器资源，也保留判分结果；不得用 synthetic 或性能 workload 替换。

### 6.2 运行与判分

能力评测采集：

```bash
cd /Users/augustus/Desktop/LLMBenchmark
python -m data_collector.cli list
python -m data_collector.cli run config/edge_router_config.yaml -y
```

并发阶梯多 Query 性能测量（另开一个窗口运行机器监控）：

```bash
python tools/monitor_machine.py --duration 3600 --interval 2   --out results/mac_benchmark/monitor/<model_tag> --chart

python tools/bench_multi_query.py   --model 'mlx-community/Qwen3.6-35B-A3B-4bit'   --port 18081   --limit-per-dataset 0   --concurrency 1,2,4,8,16   --out results/mac_benchmark/multi_query/<model_tag>
```

并发多 Query 测量时，以 `(dataset_id, record_index)` 严格对齐；每个模型共用同一批能力 query，不要与历史 882 题矩阵或其他抽样混用。

本地模型配置与入口：

```bash
# 从 config/local_model_eval.yaml 启动
/Users/augustus/miniconda3/bin/python tools/run_local_bench.py   --model-name qwen3.8-27b --limit-per-dataset 0

# 汇总并成图
/Users/augustus/miniconda3/bin/python tools/summarize_multi_query.py   --result-dir results/mac_benchmark/multi_query/qwen3.8-27b   --chart
```

### 6.3 解释边界

- 这些数据集带标准答案，可同时报告能力准确率与性能指标；
- 不同引擎、量化、并发档位仍分表分图，跨引擎只观察不直接排名；
- 失败样本保留并记录 `finish_reason` / error / timeout，不删除；
- 判分口径沿用仓库 evaluator，不另造评分逻辑。

## 7. 评测流程

| 阶段 | 操作 | 通过标准 |
|---|---|---|
| 0. 取得窗口 | 确认独占时间，记录机器基线 | 无其他推理和大文件任务 |
| 1. 环境核查 | 版本、端口、内存、swap、磁盘 | 环境符合基线 |
| 2. 资产校验 | 固定 revision、权重格式、量化、draft | 文件大小和哈希正确 |
| 3. 冒烟测试 | 短能力题加载、健康检查、一次 Chat 请求 | HTTP 2xx，模型可用 |
| 4. 冷加载记录 | 记录启动至 ready 时间、峰值资源 | 无 OOM、无反复重启 |
| 5. 正式矩阵 | 预热并按下述能力评测集与并发阶梯重复测试 | 每个测试点数据完整 |
| 6. 长题探测 | 长输入能力题单独探测并发与资源边界 | 明确记录成功或失败 |
| 7. 复核汇总 | 检查日志与资源曲线，分引擎汇总 | 异常有说明，标签完整 |

Splash 冒烟要求：`/ready` 成功、`/status` 信息正确、`/v1/chat/completions` 返回 HTTP 2xx，日志无 `Engine failed`、EOF、OOM 或重复重启。

MLX-VLM 冒烟要求：`/health` 返回 `model_ready=true`、`/v1/models` 可访问、Chat Completions 正常。

Qwen3.8 历史 Splash 日志出现过 `Engine failed` 和引擎重启，旧日志只能作参考；正式测试前必须重新 smoke 并保存新日志。

## 8. 具体实现

### 8.1 服务与端口

| 表 | 引擎 | 模型与权重 | 端口 |
|---|---|---|---:|
| A | Splash 1.5.0 | Qwen3.8-27B，GGUF Q4_K_M + DFlash2 | 18080 |
| B | Splash 1.5.0 | Qwen3.6-35B-A3B，MLX 4bit affine + DFlash2 | 18081 |
| C | MLX-VLM 0.8.0 | Gemma-4-26B-A4B，MLX 4bit affine | 18091 |
| D（第二轮） | Splash 1.5.0 | Qwen3.8-Flash-Next，低比特（IQ2/Q2 类） | 18082 |

Splash 固定 `--language-only`、`--offline`、`--kv-format int8`、`--state-format bf16`；`--queue-size` 跟随当前并发阶梯且至少 ≥ 当前并发，`--max-memory` 从 40G/42G 起按实测调整。Qwen3.8 与 Qwen3.6 的 DFlash2 不可互换。MLX-VLM 的 `--max-tokens` 是单次生成上限，不是上下文窗口；`--max-num-seqs` 按当前并发阶梯设置并记录。

表 D 仅在第二轮启动，且必须先完成权重下载与哈希校验；其 `weight_format` / `quantization` 与表 A/B/C 完全不同类别，结果只做 SSD 流式与边界行为记录，不参与任何模型间排名。

### 8.2 请求与计时

两个引擎统一通过 OpenAI 兼容接口 `/v1/chat/completions`：

- 非流式请求获取 usage、服务端耗时和完整响应；
- 流式请求记录请求开始、首个 chunk、末个 chunk，计算 TTFT 和 E2E；
- 两种口径分别存储，不混为一个指标；
- 并发档位使用多线程/异步客户端同时下发多个 query，per-request 与 per-stage 统计分开保存，队列时间单独记录；
- Prefill/Decode 优先使用服务端 timing；缺失时保留 `null`，Decode 若用流式估算必须标注方法。

### 8.3 资源采样与结果结构

同步采集推理进程 RSS、wired、swap、SSD 读写、CPU/GPU/温控状态和 error/timeout/finish_reason，并按并发档位分面保存时间序列。机器指标至少包括：CPU 使用率（占用/总核）、GPU/NPU 使用率（`powermetrics --samplers gpu_power,ane_power`；无 sudo 标“未采集”）、内存 used/free/cache/available、算力（量化单位 + TFLOPS/PFLOPS）与 GPU/NPU 带宽（理论值与实测值分列）。`powermetrics` 等命令无 sudo 权限时标记“未采集”，不得用推测值补齐。

同时生成机器指标图表：每档并发输出 CPU 使用率曲线、GPU/NPU duty/功率曲线、内存占用曲线、吞吐/算力曲线；图表与 JSON 原始采样同目录保存，图与数必须成组更新。

每轮实验保存三层 fingerprint：

```text
hardware_fingerprint：主机、芯片、内存、系统版本
engine_fingerprint：引擎、版本、端口、启动参数
model_fingerprint：模型、格式、量化、revision、draft、KV、文件哈希
```

结果先保存 JSON，再生成汇总表。建议脚本结构：

```text
tools/
  data_collector/       # 仓库原生能力评测集采集与判分入口
  bench_common.py       # 客户端计时、统计、JSON schema
  bench_splash.py       # Splash 18080/18081
  bench_mlxvlm.py       # MLX-VLM 18091
  bench_multi_query.py  # 能力集并发阶梯多 query
  monitor_machine.py    # CPU/GPU/NPU/内存/算力/带宽实时监控
  summarize.py          # 分引擎汇总、Markdown 和图表
```

## 9. 结果组织与对比原则

最终至少生成三张表，分别对应服务表 A/B/C。每行记录：上下文、cold load、TTFT、Prefill tok/s、Decode tok/s、峰值 RSS、wired、swap、SSD 读、错误数、`finish_reason`、是否成功。

**可以比较：**

- 同一引擎、同一机器、同一量化格式下的模型；
- 同一输入/输出长度和采样参数下的 TTFT、Prefill、Decode；
- 同一能力 query 长度桶与并发档位下的峰值内存和稳定性；
- 同一 warm-up 与重复策略下的统计分布。

**不应直接比较：**

- Splash 的 DFlash2 Decode 与 MLX-VLM 普通 Decode 直接排名；
- GGUF Q4_K_M 与 MLX 4bit affine 的差异解释为“模型能力差异”；
- 不同 KV 精度、上下文或 reasoning effort 混表；
- 冷启动单次结果与 warm 后多次结果混表；
- 不同并发档位混表、或把并发的吞吐/延迟与单请求指标合并；
- 只写“某模型 tok/s”而不标引擎和量化；
- 第二轮 Flash-Next（低比特/流式口径）与第一轮任何模型直接排名。

本轮只在一台 M5 Pro 上做单机摸底，不将结果外推为 M4、M5、M5 Pro 的芯片排名。

最终至少四张表：表 A/B/C（第一轮）+ 表 D（第二轮，实验性单独标注）。

## 10. 性能归因方法

| 因素 | 主要影响 | 测试观察 |
|---|---|---|
| 内存带宽 | Decode 通常最敏感，Prefill 也受影响 | 同模型、同引擎、同上下文比较 tok/s |
| GPU 核数与架构 | Prefill 主要瓶颈之一，Decode 中高 | 单机内不同模型和上下文对比 |
| 统一内存 | 决定 KV Cache、Metal 缓冲和上下文上限 | 观察阶梯中的 OOM、swap 和 wired 增长 |
| KV 精度 | 影响内存和速度 | Splash 全程固定 int8，其他引擎记录实际配置 |
| DFlash2 | 主要影响 Decode | 表 A/B 单独标注 draft，不跨模型复用 |
| 专家权重驻留 | MoE 模型的 Decode 与内存关键 | 记录 `--expert-memory` 和 SSD 读取 |
| SSD 流式读取 | 冷加载和 swap 时影响很大 | `iostat` 与加载日志对照 |
| 温控与功耗 | 长时间运行可能降频 | 比较首轮与末轮，必要时记录温度/功耗 |

归因报告需回答：Prefill 随长度如何变化；Decode 与内存带宽、权重格式和 DFlash2 的关系；KV Cache、权重和 Metal 缓冲各占多少；上下文失败发生在加载、Prefill 还是 Decode；是否出现 SSD 流式读取、swap 或热降频；重复测试的离散程度和最差结果。

## 11. 执行计划、耗时与验收

| 场景 | 预留时间 |
|---|---:|
| 仅加载与接口冒烟 | 15～30 分钟 |
| 单模型快速摸底 | 1.5～2 小时 |
| 单模型正式矩阵 | 4 小时 |
| 首次执行或含长上下文能力题探测 | 6 小时 |
| 三个模型正式评测 | 12～18 小时，拆成 3 个独立窗口 |
| 第二轮资产下载校验（Flash-Next 低比特） | 依权重体积另计，必须独立于测速窗口 |
| 第二轮 Flash-Next 缩减矩阵 | 3～4 小时（短-中能力题并发探测 + 长题探测，预留失败重试） |

推荐顺序：

```text
窗口 1：Qwen3.8-27B（Splash，建立基线）
窗口 2：Qwen3.6-35B-A3B（Splash）
窗口 3：Gemma-4-26B-A4B（MLX-VLM）
（第一轮报告完成后，经确认再启动第二轮）
窗口 4：Qwen3.8-Flash-Next 低比特权重下载、校验与冒烟
窗口 5：Flash-Next 能力题缩减矩阵 + 长题探测（实验性，单独标注）
```

模型之间至少留 15～30 分钟冷却和残留进程检查。连续满载十几小时会引入热降频、内存碎片和系统缓存差异。

**验收标准：**

- [ ] 能力评测集数据文件存在且 `(dataset_id, record_index)` 对齐；判分结果可追溯；
- [ ] 环境、模型、引擎和参数 fingerprint 完整；
- [ ] smoke test 通过，日志无 OOM/EOF/重复重启；
- [ ] 每个成功测试点至少 3 次有效重复；
- [ ] 报告均值、中位数、P95 和最差值；
- [ ] 每个能力数据集与并发档位是否稳定通过有明确记录；
- [ ] 长输入能力题探测单独标记，不与常规能力评测结果混算；
- [ ] 每个并发档位的 CPU/GPU/NPU、内存、算力、带宽原始采样与图表可追溯；
- [ ] Prefill、Decode、内存、swap、SSD 数据可追溯；
- [ ] 失败样本保留并注明原因；
- [ ] 分引擎输出表格和图表，不合并排名；
- [ ] 第一轮（表 A/B/C）报告完成后才启动第二轮；
- [ ] 表 D 单独标注实验性口径，不与表 A/B/C 排名。

## 12. 主要风险与最终建议

| 风险 | 级别 | 处理方式 |
|---|---|---|
| 共享服务未释放 | 高 | 先协调独占窗口，不擅自停止同事服务 |
| Qwen3.8 历史 Splash 重启 | 中高 | 正式前重新 smoke，异常则不进入矩阵 |
| 双引擎结果被误读 | 中高 | 强制分表，跨引擎只做观察性描述 |
| 长输入能力题 OOM/swap | 中高 | 单独探测，不与常规能力评测结果混表 |
| GGUF Q4_K_M 与 MLX 4bit 混用 | 中 | 每行记录权重格式和量化 |
| 下载或写盘污染测速 | 中 | 测试窗口内禁止下载、复制和索引构建 |
| 4bit/Q4 被误读为质量上限 | 中 | 报告明确限制，能力评测另立项目 |
| Flash-Next 资产未就绪 | 中高 | 第二轮启动前必须完成下载与哈希校验，未就绪则推迟 |
| 低比特权重被误读为模型质量 | 中 | 表 D 明确标注 IQ2/Q2 实验口径，不代表模型真实能力 |
| Flash-Next 频繁 OOM / 流式读取 | 中高 | 缩减矩阵按计划执行，失败也是有效边界数据 |

**建议批准第一轮执行。** 三个模型的资产和推理环境均已具备，主测机能够覆盖本轮文本模型性能摸底；但独占窗口是硬前置条件，不是优化项。第二轮 Qwen3.8-Flash-Next 为实验性探测：权重资产尚需下载校验，Q4 体积超出 64GB 内存，只能走低比特 / SSD 流式口径，建议作为独立事项在第一轮报告完成后另行确认。

最终交付口径为：Qwen3.8-27B 建立 Splash 基线；Qwen3.6-35B-A3B 使用已校验 DFlash2 完成同一矩阵；Gemma-4-26B-A4B 使用 MLX-VLM 单独成表；提交“分引擎性能表 + 内存/上下文曲线 + 失败边界 + 硬件归因”，不提交跨引擎总排名。第二轮若执行，交付物为表 D（Flash-Next 低比特缩减矩阵 + SSD 流式 / 边界行为记录），单列实验性口径。
