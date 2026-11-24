# MLX 文本模型与 Qwen-Image 生图模型运行时规划

> 目标机器：`192.168.3.5` / `macmini-m5-jiuwen` / Apple M5 Pro / 64GB 统一内存  
> 当前阶段：单模型性能摸底，不做路由评测  
> 后续生图目标：Qwen-Image 20B  
> 核查日期：2026-11-23（Asia/Shanghai）

---

## 1. 结论先行

### 1.1 三个“文本模型”仍然需要 mlx-vlm

虽然本轮只输入文本，但三个目标模型都是多模态条件生成架构，不能按普通纯文本模型处理：

| 模型 | 架构 | 推荐运行时 |
|---|---|---|
| Qwen3.8-27B | `Qwen3_5ForConditionalGeneration` | `mlx-vlm` |
| Qwen3.6-35B-A3B | `Qwen3_5MoeForConditionalGeneration` | `mlx-vlm` |
| Gemma-4-26B-A4B | `Gemma4ForConditionalGeneration` | `mlx-vlm` |

因此：

- 文本输入也应走 `mlx-vlm` 的 text-only 路径；
- 不要用 `mlx-lm` 的结果替代 `mlx-vlm`；
- 三个模型的性能结果必须来自同一个运行时版本和同一套评测脚本；
- 当前 3.5 已经安装 `mlx-vlm 0.8.0`，本轮无需再新增配置。

### 1.2 Qwen-Image 20B 不使用 mlx-vlm

Qwen-Image 20B 是扩散式文本到图像模型，不是自回归视觉语言模型：

- 它包含 DiT/Transformer 主干、VAE 和文本编码器；
- 生成过程是扩散去噪，不是逐 token Decode；
- `mlx-vlm.generate` 的 Prefill/Decode 计时口径不适用于它；
- 应使用支持 Apple Silicon MLX 的扩散生图运行时。

当前建议：

```text
文本/多模态理解模型：mlx-vlm
Qwen-Image 20B 生图：mflux
```

> 注意：mflux 的 Qwen-Image 实现内部会使用 Qwen2.5-VL 作为文本编码器，但这不等于用 `mlx-vlm` 直接生成图片。模型图、调度器和生成入口都不同。

---

## 2. 当前 3.5 环境实机核查

已在 `192.168.3.5` 的独立环境 `/Users/jiuwen/mlx-bench-venv` 中确认：

| 包 | 当前状态 |
|---|---|
| Python | 3.14.8 |
| mlx | 0.32.4 |
| mlx-metal | 0.32.4 |
| mlx-vlm | **0.8.0，已安装** |
| transformers | 5.24.0 |
| mlx-lm | 未安装 |
| torch | 未安装 |
| diffusers | 未安装 |
| mflux | 未安装 |

可执行文件已存在：

```text
/Users/jiuwen/mlx-bench-venv/bin/mlx_vlm.generate
/Users/jiuwen/mlx-bench-venv/bin/mlx_vlm.server
/Users/jiuwen/mlx-bench-venv/bin/mlx_vlm.chat
```

### 2.1 结论

- 文本三模型的运行时已经准备好；
- 现在不需要为了文本评测再安装 `mlx-vlm`；
- 当前也不应在 `mlx-bench-venv` 中安装 `mflux`、`torch` 或 `diffusers`。

---

## 3. 为什么 mflux 要单独建环境

mflux 和 mlx-vlm 虽然都依赖 MLX，但依赖面和评测口径不同：

| 维度 | mlx-vlm | mflux |
|---|---|---|
| 任务 | 多模态理解、文本生成 | 扩散式图像生成 |
| 生成方式 | 自回归 token 生成 | 多步去噪 |
| 关键指标 | Prefill、Decode、TTFT | 每步耗时、总生图耗时、峰值内存 |
| 依赖 | transformers、tokenizer | torch、diffusion 相关组件等 |
| 评测环境 | 文本基线环境 | 生图独立环境 |

建议建立两个环境：

```text
~/mlx-bench-venv      # 已存在，只用于 mlx-vlm 文本/多模态评测
~/mflux-bench-venv    # 后续新建，只用于 Qwen-Image 20B 生图评测
```

这样可以避免：

- mflux 安装时升级 `mlx`、`transformers` 等包，改变文本基线；
- torch 和其他依赖影响文本模型加载时间与内存；
- 两个阶段的版本锁文件混在一起，导致结果不可复现。

---

## 4. Qwen-Image 20B 的机器选择

### 4.1 推荐机器

| 机器 | 统一内存 | 建议 |
|---|---:|---|
| `192.168.3.5` M5 Pro | 64GB | **首选**，当前文本主测机，资源相对干净 |
| `192.168.3.7` M4 Pro | 64GB | 可作为 M4/M5 对照机，但需先隔离 Ollama/oMLX/EXO 等负载 |
| `192.168.3.13` M5 Pro | 64GB | 不建议作为主测，后台服务会污染结果 |
| 48GB Mac mini | 48GB | **不建议作为 8bit 正式基线**，余量偏小，容易 swap |

### 4.2 内存粗估

Qwen-Image 20B 的 20B 主要指扩散 Transformer 参数量，此外还有文本编码器和 VAE：

```text
20B 扩散主干（8bit）      约 20GB
Qwen2.5-VL 7B 文本编码器  约 14GB（mflux 当前实现通常保留 bfloat16）
VAE、激活值、运行时开销    数 GB
```

因此：

- 8bit 总占用粗估约 **35～45GB**，分辨率越高、步数越多，激活和临时缓存越大；
- 64GB 机器可以尝试 8bit 单任务，但正式测试必须记录峰值内存和 swap；
- 48GB 机器运行 8bit 全流程风险较高，不建议作为质量基线；
- 4bit 可以降低主干权重，但 mflux 文档明确提示 Qwen-Image 在 6bit 及以下可能出现明显画质退化，不能把 4bit 质量结果与 8bit 混在一起。

> 以上是预算估算，不是实测结论。最终必须以固定分辨率、固定步数下的峰值统一内存和 swap 为准。

---

## 5. 量化策略

### 5.1 推荐分两阶段

| 阶段 | 量化 | 目的 |
|---|---|---|
| 性能摸底 | 4bit 或 8bit 均可 | 观察不同量化下的速度、内存和稳定性 |
| 质量基线 | **8bit 优先** | 尽量接近参考画质，避免把量化退化误判为模型能力 |

### 5.2 结果标注要求

每条结果必须写明：

```text
model_repo
revision
quantization
text_encoder_precision
resolution
steps
guidance
seed
negative_prompt
runtime=mflux
mflux_version
mlx_version
macOS_version
machine_model
peak_memory
swap_used
```

不能只写“Qwen-Image 20B”，否则 4bit/8bit、不同分辨率和不同步数的结果不可比较。

---

## 6. 生图评测指标需要重新设计

文本模型的 Prefill、Decode、TTFT 不能直接套到扩散生图上。建议改为：

### 6.1 性能指标

- 模型加载时间；
- 文本编码/提示词嵌入耗时；
- 单步去噪耗时，记录 p50、p95；
- 固定步数下的总生图耗时；
- 每秒生成图片数或每分钟图片数；
- 首张图片完成时间；
- MLX 峰值内存；
- 进程最大 RSS；
- swap 使用量；
- 不同分辨率下的扩展曲线；
- 不同步数下的扩展曲线；
- 失败率和 OOM 次数。

### 6.2 质量指标

- 固定 prompt 集的提示词遵循度；
- 中英文文字渲染正确率；
- 主体、数量、空间关系正确率；
- 画面审美与细节；
- 固定 seed 下的可复现性；
- 4bit 与 8bit 的并排对比；
- 必要时增加人工评分或统一的多模态裁判，但不能用不同模型自评替代人工基准。

### 6.3 推荐测试矩阵

第一轮不要铺太大，建议：

```text
分辨率：512x512、768x768、1024x1024
步数：10、20、30
batch size：1
seed：固定 42
CFG/guidance：固定默认值
prompt：固定 20～30 条，覆盖中文、英文、文字渲染、人物、物体数量、复杂场景
```

质量基线单独使用：

```text
1024x1024、30 steps、8bit、固定 seed 和固定 negative prompt
```

---

## 7. 与 Qwen-Image 20B 相关的执行顺序

### 阶段 A：先完成文本模型下载和基线

当前 3.5 正在顺序下载：

1. `Qwen3.6-35B-A3B-4bit`
2. `Gemma-4-26B-A4B-4bit`

要求：

- 严格串行；
- 下载期间不安装 mflux；
- 下载期间不跑正式生图；
- 下载完成并校验后再进入文本模型评测。

### 阶段 B：完成文本模型评测

使用现有：

```text
/Users/jiuwen/mlx-bench-venv/bin/mlx_vlm.generate
```

先完成 Qwen3.8-27B、Qwen3.6-35B-A3B、Gemma-4-26B-A4B 的 Prefill、Decode、TTFT、峰值内存和上下文测试。

### 阶段 C：建立 mflux 独立环境

在文本基线完成后再执行，避免依赖变化污染已有结果。建议命令骨架：

```bash
/opt/homebrew/bin/python3.14 -m venv ~/mflux-bench-venv
~/mflux-bench-venv/bin/python -m pip install --upgrade pip setuptools wheel
~/mflux-bench-venv/bin/python -m pip install mflux==0.24.0
```

安装后先记录：

```bash
~/mflux-bench-venv/bin/python -m pip freeze > ~/mflux-bench-venv/requirements-lock.txt
~/mflux-bench-venv/bin/mflux-generate-qwen --help
```

> 执行前应再次确认 mflux 最新稳定版本、Python 版本和 macOS 兼容性，不要直接复用本文档中的版本号而不做检查。

### 阶段 D：最小生图验证

先用小分辨率、少步数验证链路：

```text
512x512
4 steps
1 张图
固定 prompt
```

确认以下内容后再跑正式矩阵：

- 模型权重能加载；
- 文本编码器可用；
- VAE 能解码；
- 输出图片可以写入本地；
- 内存没有立即触发 OOM；
- swap 没有异常增长。

---

## 8. 运行时选择的最终判断

| 需求 | 运行时 |
|---|---|
| Qwen3.8-27B 文本/多模态理解 | `mlx-vlm` |
| Qwen3.6-35B-A3B 文本/多模态理解 | `mlx-vlm` |
| Gemma-4-26B-A4B 文本/多模态理解 | `mlx-vlm` |
| Qwen-Image 20B 文本生图 | `mflux` |
| Qwen-Image 20B 图片编辑 | mflux 的 Qwen-Image Edit 路径，需单独确认模型和权重 |

---

## 9. 当前行动建议

1. **不要现在安装 mflux。** 当前 3.5 正在下载 Qwen3.6，随后还要下载 Gemma；先保证下载和文本基线不被干扰。
2. **保留现有 `mlx-bench-venv`。** 它已经满足三个文本模型的 `mlx-vlm` 运行需求。
3. **后续为 Qwen-Image 单独建立 `mflux-bench-venv`。**
4. **Qwen-Image 首轮优先在 64GB 机器上做 8bit 单任务。**
5. **不要把 Prefill/Decode 指标直接用于生图模型。** 改为每步耗时、总耗时、吞吐和峰值内存。
6. **4bit 和 8bit 分开报告。** 不能只写模型名，必须写量化、分辨率、步数和随机种子。

---

## 10. 一句话总结

> 三个文本模型继续统一使用已经安装好的 `mlx-vlm`；Qwen-Image 20B 另建 mflux 环境、另设生图指标、另做 8bit 质量基线，二者不要混在同一个虚拟环境和同一套 Prefill/Decode 口径里。
