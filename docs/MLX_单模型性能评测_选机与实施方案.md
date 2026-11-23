# MLX 单模型性能评测：选机、部署与实施方案

> 目标仓库：`Augustus-Lee/LLMRouterBench`（本地 `/Users/augustus/Desktop/LLMBenchmark`）  
> 评测范围：**只做单模型性能评测，不做路由，不做质量榜单**  
> 推理框架：**MLX（统一使用 mlx-vlm 运行时）**  
> 目标模型：Qwen3.8-27B、Qwen3.6-35B-A3B、Gemma-4-26B-A4B  
> 核查日期：2026-11-23（Asia/Shanghai）

---

## 1. 结论先行

### 1.1 推荐机器

| 角色 | 机器 | 推荐结论 |
|---|---|---|
| **主测机（首选）** | `192.168.3.5` / `macmini-m5-jiuwen` / M5 Pro 64GB | **推荐。** 资源最干净、磁盘余量充足、已有 Qwen3.8-27B MLX 权重，适合建立统一基线并承载三个模型 |
| **对照机（可选）** | `192.168.3.7` / `macmini-m4-admin` / M4 Pro 64GB | **作为 M4 vs M5 硬件对照。** 有 3TB 磁盘余量，但当前运行 Ollama、oMLX、EXO 等服务，需先隔离负载 |
| **不推荐做主测** | `192.168.3.13` / `macmini` / M5 Pro 64GB | **排除主测。** Splash 常驻推理服务实测单引擎 RSS 约 12.4GiB（历史服务总占用更高）、磁盘仅约 73GiB，会污染性能结果 |

### 1.2 一句话方案

> **把 `192.168.3.5` 作为唯一的正式主测机；在独立虚拟环境中用 `mlx-vlm` 依次加载三个 4bit 模型，固定 prompt、生成长度、上下文和采样参数，重复测量 Prefill、Decode、TTFT、峰值统一内存与实际可用上下文。`192.168.3.7` 只用于同权重、同运行时的 M4 对照复测。**

### 1.3 每个目标模型放在哪台机器

| 目标模型 | 主测机 | 对照机 | 当前资产状态 |
|---|---|---|---|
| Qwen3.8-27B | **192.168.3.5** | 192.168.3.7（需补权重） | 3.5 已有 MLX 4bit，已完成 `mlx-vlm` 冒烟 |
| Qwen3.6-35B-A3B | **192.168.3.5** | 192.168.3.7（需补权重） | 3.5 待下载约 19.0GiB MLX 4bit |
| Gemma-4-26B-A4B | **192.168.3.5** | 192.168.3.7（已有 oMLX 目录，需标准化） | 3.5 待下载约 14.3GiB MLX 4bit |

> 统一原则：三个模型在同一台机器、同一个 `mlx-vlm` 环境中完成首轮测试；只有需要分析 M4/M5 硬件差异时，才在 3.7 复跑代表性场景。

### 1.4 必须先纠正的技术点

**MLX ≠ 只能是 `mlx-lm`。** 三个目标模型的配置和模型卡均表明它们属于多模态 `*ForConditionalGeneration` 架构，官方转换与 README 指定使用 **`mlx-vlm`**：

| 模型 | architecture | model_type | 指定运行时 |
|---|---|---|---|
| Qwen3.8-27B | `Qwen3_5ForConditionalGeneration` | `qwen3_5` | `mlx-vlm` |
| Qwen3.6-35B-A3B | `Qwen3_5MoeForConditionalGeneration` | `qwen3_5_moe` | `mlx-vlm` |
| Gemma-4-26B-A4B | `Gemma4ForConditionalGeneration` | `gemma4` | `mlx-vlm` |

因此：

- 不应把 `mlx_lm.generate` 加载失败直接解释为“模型或机器不支持”；
- 本阶段应统一使用 **`mlx-vlm`**，即使输入是纯文本，也走其 text-only 调用路径；
- `mlx-lm` 仅作为排查工具，不与 `mlx-vlm` 结果混为同一性能基线；
- 三个模型必须使用相同运行时版本、相同量化类型和相同评测脚本。

---

## 2. 三台机器实测对比

### 2.1 硬件与资源快照

| 项目 | 192.168.3.5 Jiuwen | 192.168.3.7 Admin | 192.168.3.13 SynCo |
|---|---|---|---|
| 芯片 | **Apple M5 Pro** | Apple M4 Pro | Apple M5 Pro |
| CPU | 18 核（6 Super + 12 Performance） | 14 核（10 Performance + 4 Efficiency） | 18 核（6 Super + 12 Performance） |
| GPU | 20 核 | 20 核 | 20 核 |
| 统一内存 | **64GB** | **64GB** | **64GB** |
| macOS | 27.2 | 26.6.2 | 27.0 |
| 根卷可用空间 | **约 489GiB** | **约 3.0TiB** | **约 73GiB** |
| 当前负载（1/5/15 分钟） | 约 1.35 / 1.43 / 1.43 | 约 1.52 / 1.28 / 1.28 | 约 0.22 / 0.64 / 0.95 |
| 后台大模型服务 | 无大模型常驻 | Ollama、oMLX、EXO 等 | **Splash 常驻 Qwen3.8 GGUF** |
| 主测适配度 | **最高** | 中（需清理） | 低（干扰大） |

### 2.2 为什么首选 192.168.3.5

1. **硬件代际更新。** M5 Pro 的 CPU/GPU 代际、内存子系统和新版 Metal 更适合作为 Mac 端 MLX 主基线。
2. **内存状态干净。** 复核时空闲内存约 93%，无大模型常驻，测量结果更接近空载基线。
3. **磁盘容量足够。** 约 489GiB 可用空间，三套 4bit 权重总计约 50GiB；即使保留日志、缓存和结果，也没有压力。
4. **已有 Qwen3.8-27B 权重。** 可先零下载完成运行时与计时链路验证，再下载另外两个模型。
5. **已有 MLX 基础环境。** 当前有 Homebrew `mlx 0.32.3`、`mlx-lm 0.32.0`，但需另建隔离环境安装 `mlx-vlm`，避免污染现有运行环境。
6. **网络位置和 SSH 已打通。** 已配置 `macmini-m5-jiuwen` 别名，适合远程执行和结果回收。

### 2.3 为什么排除 192.168.3.13 做主测

- Splash 正在常驻加载 Qwen3.8-27B GGUF，单个引擎进程 RSS 约 12.4GiB，服务总占用更高；
- 当前内存空闲约 32%，且已使用 swap，无法作为干净性能基线；
- 根卷仅约 73GiB 可用，新增两个模型后余量偏紧；
- 其 M5 Pro 硬件与 3.5 相同，作为主测不会带来额外硬件信息，却会引入后台负载变量；
- 可作为“已有服务环境下”的部署验证机，但不应进入正式对照数据。

### 2.4 192.168.3.7 的定位

适合做 **M4 Pro vs M5 Pro 对照机**，原因是：

- 同样 64GB 统一内存，能够运行三个 4bit 模型；
- 有约 3.0TiB 磁盘，可保存多份权重和结果；
- 已有 Gemma-4-26B-A4B 的 oMLX 本地权重，可先验证文件完整性，但正式测试仍应从统一的 `mlx-community` 源重新部署；
- 当前 Ollama、oMLX、EXO 等多服务并存，正式测试前必须停止无关模型加载，记录后台进程，并在每轮前复核内存压力。

**结论：主测只用 3.5；需要硬件归因时，再用 3.7 做受控对照。**

---

## 3. 三个模型的 MLX 资产与内存预算

### 3.1 统一资产口径

| 模型 | 推荐 MLX 仓库 | 权重文件 | 权重实际大小 | 量化 |
|---|---|---|---:|---|
| Qwen3.8-27B | `mlx-community/Qwen3.8-27B-4bit` | 3 个 safetensors | **约 14.95GiB** | affine 4bit，group size 64 |
| Qwen3.6-35B-A3B | `mlx-community/Qwen3.6-35B-A3B-4bit` | 4 个 safetensors | **约 19.0GiB** | 主体 4bit，部分门控层 8bit |
| Gemma-4-26B-A4B | `mlx-community/gemma-4-26b-a4b-it-4bit` | 3 个 safetensors | **约 14.29GiB** | 主体 4bit，router 等敏感层 8bit |

> “权重实际大小”按 safetensors 字节数计算，不等于 `du` 显示的目录大小。已有目录可能包含 Git LFS 缓存或重复副本，例如 3.5 的 `Qwen3.8-27B-4bit` 目录显示约 30GB，但有效权重约 14.95GiB。

### 3.2 64GB 统一内存是否够

**三个模型都可以在 64GB 机器上运行**，但不能只按“权重 < 64GB”判断：

```text
可用预算
= 64GiB 物理统一内存
- macOS 与服务常驻
- 模型权重
- KV Cache
- 临时激活 / Metal 缓冲
- 输出与请求并发余量
```

建议预留：

- 系统与桌面：8～12GiB；
- 模型权重：15～19GiB；
- KV Cache 与运行时：8～24GiB（取决于上下文长度）；
- 安全余量：至少 6～10GiB，避免 swap。

因此：

- **短上下文（1K～8K）**：三个模型都有较大余量，适合稳定基线；
- **中长上下文（16K～64K）**：需要实测 KV Cache 增长和峰值内存；
- **128K 以上**：不能仅凭模型配置的 262,144 token 上限承诺可运行，必须逐级压力测试并记录失败点；
- **正式结果以“实际可运行上下文”而非模型卡最大上下文为准。**

### 3.3 重要模型差异

| 模型 | 结构特点 | 对性能的影响 |
|---|---|---|
| Qwen3.8-27B | 27B 级、64 层、线性注意力 + 全注意力混合、视觉塔 | 参数读取量较大，Decode 更受内存带宽影响；Prefill 受 GPU 矩阵吞吐影响 |
| Qwen3.6-35B-A3B | 256 experts，每 token 激活 8 个，MoE | 总权重约 19GiB，但单 token 激活计算量较低；权重读取、专家调度和内存局部性影响明显 |
| Gemma-4-26B-A4B | 128 experts，每 token 激活 8 个，滑动窗口 + 全注意力混合 | 权重约 14.3GiB，短上下文可能很快；长上下文受 KV Cache 与混合注意力实现影响 |

> 不能仅按“27B / 35B / 26B”做线性推断，MoE 的激活参数、专家路由和注意力结构会显著改变 Prefill/Decode 曲线。

---

## 4. 统一运行环境设计

### 4.1 版本基线

复核时：

- `mlx-vlm` PyPI 最新版本：**0.8.0**
- `mlx-vlm 0.8.0` 要求 `mlx>=0.32.4`、`transformers>=5.14.0`
- 3.5 原有 Homebrew `mlx`：0.32.3
- 3.5 原有 Homebrew `mlx-lm`：0.32.0
- Python：3.14.8

因此必须建立独立 venv，让 pip 安装满足 `mlx-vlm` 要求的依赖，**不要直接升级或覆盖 Homebrew 的 `mlx-lm` libexec 环境**。

### 4.2 在 192.168.3.5 创建隔离环境

```bash
ssh macmini-m5-jiuwen

/opt/homebrew/bin/python3.14 -m venv ~/mlx-bench-venv
~/mlx-bench-venv/bin/python -m pip install --upgrade pip setuptools wheel
~/mlx-bench-venv/bin/python -m pip install "mlx-vlm==0.8.0"
```

若默认 PyPI 较慢，可使用已配置的镜像：

```bash
~/mlx-bench-venv/bin/python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "mlx-vlm==0.8.0"
```

### 4.3 锁定与记录版本

安装后执行：

```bash
~/mlx-bench-venv/bin/python - <<'PY'
import mlx, mlx_vlm, transformers, huggingface_hub
print("mlx", mlx.__version__)
print("mlx_vlm", mlx_vlm.__version__)
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
PY

# mlx 新版可能不导出 __version__，上面的脚本若因此失败，
# 应改用 importlib.metadata.version("mlx") 读取。
~/mlx-bench-venv/bin/python -m pip freeze > ~/mlx-bench-venv/requirements-lock.txt
```

正式评测报告中必须记录 `requirements-lock.txt` 的哈希或完整内容。

### 4.4 最小加载验证（先跑已有 Qwen3.8）

先确认 CLI 与模型加载链路：

```bash
~/mlx-bench-venv/bin/mlx_vlm.generate \
  --model ~/Models/Qwen3.8-27B-4bit \
  --prompt "用一句话解释什么是统一内存。" \
  --max-tokens 32 \
  --temperature 0.0
```

若该命令成功：

- 说明 `mlx-vlm`、Metal、权重和 tokenizer 链路可用；
- 再下载另外两个模型；
- 不要跳过此步直接开始长时间批量评测。

---

## 5. 模型部署与下载

### 5.1 部署目录规范

在 3.5 上统一使用：

```text
~/Models/Qwen3.8-27B-4bit                 # 已有
~/Models/Qwen3.6-35B-A3B-4bit             # 待下载
~/Models/gemma-4-26b-a4b-it-4bit          # 待下载
```

下载完成后必须检查：

```bash
for m in \
  ~/Models/Qwen3.8-27B-4bit \
  ~/Models/Qwen3.6-35B-A3B-4bit \
  ~/Models/gemma-4-26b-a4b-it-4bit
do
  echo "## $m"
  find "$m" -maxdepth 1 -name '*.safetensors' -exec stat -f '%z %N' {} \; \
    | awk '{s+=$1; n++} END {printf "safetensors=%.2f GiB files=%d\n", s/1073741824, n}'
  test -s "$m/config.json" && echo "config OK"
  test -s "$m/tokenizer_config.json" && echo "tokenizer OK"
done
```

### 5.2 ModelScope 下载入口

三个仓库均可从 ModelScope 获取：

```text
mlx-community/Qwen3.8-27B-4bit
mlx-community/Qwen3.6-35B-A3B-4bit
mlx-community/gemma-4-26b-a4b-it-4bit
```

推荐安装 `modelscope` 或使用支持断点的下载方式：

```bash
~/mlx-bench-venv/bin/python -m pip install modelscope
~/.local/bin/modelscope download \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --local_dir ~/Models/Qwen3.6-35B-A3B-4bit

~/.local/bin/modelscope download \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --local_dir ~/Models/gemma-4-26b-a4b-it-4bit
```

如果下载中断：

- 保留已完成的 `.safetensors`；
- 重新执行同一条命令，利用断点续传；
- 不要删除整个目录后反复从头下载；
- 下载完成后核对 safetensors 文件数量与字节数。

### 5.3 网络建议

此前实测：

- 3.5 和 3.7 当前主要走 Wi-Fi，Ethernet `en0` 未启用；
- 3.5 Wi-Fi 协商速率约 585Mbps，但实际互联网吞吐明显更低；
- Hugging Face 连接曾多次重置；
- ModelScope 可用且支持 Range，但单连接约 0.5～1.2MB/s，4 路并发约 2.1MB/s。

建议：

1. 优先接有线 Ethernet；
2. 使用 ModelScope 并保留断点；
3. 避开共享网络高峰；
4. 不要在下载期间同时进行正式性能测试；
5. 下载任务与推理任务分离，避免 SSD 和网络 I/O 污染结果。

### 5.4 关于已存在的 oMLX Gemma 目录

3.7 上的：

```text
~/.omlx/models/gemma-4-26b-a4b-it-4bit
```

包含 MLX safetensors，可作为“权重存在”的证据，但它位于 oMLX 应用目录，可能含应用私有元数据（如 `.msc`、`.mv`），**不建议直接作为统一评测部署目录**。

正式方案应：

- 从 `mlx-community/gemma-4-26b-a4b-it-4bit` 重新建立标准目录；
- 记录 revision；
- 校验文件数量和大小；
- 不混用 oMLX 的推理路径与 `mlx-vlm` 的结果。

---

## 6. 正式评测设计

### 6.1 测试矩阵

#### 维度一：输入上下文阶梯

| 场景 | Prompt tokens | 目的 |
|---|---:|---|
| S1 | 128 | 最短链路、固定开销 |
| S2 | 1K | 常规对话 |
| S3 | 4K | 中等文档 |
| S4 | 8K | 长文档起点 |
| S5 | 16K | 长上下文压力 |
| S6 | 32K（可选） | 长上下文扩展 |
| S7 | 64K（可选） | 接近实用上限 |
| S8 | 128K（仅稳定性） | 失败点/内存边界 |

> 不建议一开始就对三个模型执行 262K 满上下文压力测试。先用阶梯找到稳定边界，再单独做上限探测。

#### 维度二：输出长度

- 固定 `max_tokens=256` 为主；
- 另做 `max_tokens=512` 复核 Decode 稳定性；
- 不让模型自由决定生成长度，否则无法比较。

#### 维度三：重复次数

- 每个场景预热 1 次，丢弃；
- 正式测量至少 3 次；
- 报告 median，同时记录 min/max 与标准差；
- 机器负载异常时整组重跑，不只挑最快值。

### 6.2 固定参数

```text
temperature = 0
top_p = 1.0（或统一固定值）
max_tokens = 256
seed = 42
batch_size = 1
concurrency = 1
```

说明：

- `temperature=0` 用于降低采样随机性，不是性能加速手段；
- 不启用 speculative decoding、draft model、prompt cache；
- 不在不同模型间改变 KV Cache 量化策略；
- 如需测试 KV Cache 量化，应作为单独实验组。

### 6.3 指标定义

#### Prefill / TTFT

```text
TTFT = 首个输出 token 的时间 - 请求开始时间
Prefill throughput = prompt_tokens / TTFT
```

报告时同时给出：

- `TTFT_ms`
- `prefill_tokens_per_sec`
- prompt token 实际数量

#### Decode

```text
Decode throughput = (output_tokens - 1) / (last_token_time - first_token_time)
```

报告：

- `decode_tokens_per_sec`
- 输出 token 数
- 首 token 到末 token 的实际耗时

#### 峰值内存

同时记录三类值：

1. **MLX 自报峰值**：使用 `mlx.core.get_peak_memory()`；旧接口 `mlx.core.metal.get_peak_memory()` 仍可用，但会给出弃用警告；
2. **进程峰值 RSS / footprint**：通过 `psutil` 或 `ps` 定时采样；
3. **系统压力**：`memory_pressure`、`vm_stat`、`sysctl vm.swapusage`。

三者不能互相替代：

- MLX 峰值反映框架分配；
- RSS 反映进程占用；
- swap 反映是否已经超出物理内存。

#### 最大可用上下文

采用二分或阶梯法：

```text
1K → 2K → 4K → 8K → 16K → 32K → 64K → 128K
```

判定成功必须同时满足：

- 请求完成且输出有效；
- 无 OOM；
- 无显著 swap 增长；
- 进程未崩溃；
- 延迟没有出现数量级异常。

记录：

- 最大成功 prompt tokens；
- 首次失败点；
- 失败类型（OOM / timeout / Metal error / 进程退出）；
- 对应峰值内存。

### 6.4 每轮必须记录的元数据

```text
机器：hostname、芯片、CPU/GPU 核数、统一内存、macOS 版本
运行时：mlx、mlx-vlm、transformers、Python 版本
模型：仓库、revision、量化方式、safetensors 数量与总字节数
请求：prompt tokens、max_tokens、temperature、seed、上下文限制
结果：TTFT、Prefill、Decode、峰值内存、swap、成功/失败
环境：后台进程、内存压力、网络/磁盘活动、热状态
```

### 6.5 结果表模板

| model | quant | prompt_tokens | max_tokens | TTFT_ms | prefill_tok/s | decode_tok/s | peak_mem_GiB | swap_MB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.8-27B | 4bit | 128 | 256 |  |  |  |  |  |  |
| Qwen3.6-35B-A3B | 4bit | 128 | 256 |  |  |  |  |  |  |
| Gemma-4-26B-A4B | 4bit | 128 | 256 |  |  |  |  |  |  |

每个上下文长度单独一张表，或在同一表中增加 `context` 列。

---

## 7. 机器性能对推理速度的影响

### 7.1 主要影响因素排序

1. **统一内存带宽**
   - Decode 每生成一个 token 都要读取模型权重；
   - 对单请求 Decode，内存带宽通常比 CPU 核数更关键；
   - MoE 模型还受专家权重访问局部性影响。

2. **GPU 核心数与代际**
   - Prefill 是矩阵密集型阶段，主要受 GPU 计算吞吐、Metal 内核和矩阵单元影响；
   - 相同 GPU 核数不代表相同代际性能，M4 Pro 与 M5 Pro 必须实测。

3. **统一内存容量与可用余量**
   - 容量决定可加载的权重与 KV Cache；
   - 余量不足会触发 swap，导致延迟剧烈抖动；
   - 64GB 是“能跑”与“稳定跑长上下文”的分界线，不是统一答案。

4. **上下文长度**
   - 输入越长，Prefill 计算和 KV Cache 越大；
   - 长上下文可能使内存带宽、缓存和注意力实现成为主瓶颈；
   - 不同模型的滑动窗口/线性注意力结构会导致曲线不同。

5. **CPU 与系统调度**
   - tokenizer、采样、请求调度、模型加载会使用 CPU；
   - 后台应用和其他模型服务会抢占 GPU/内存带宽；
   - 共享机器必须记录并隔离负载。

6. **散热与持续负载**
   - 短跑和持续 10 分钟压力测试可能不同；
   - 应记录频率/温度/性能警告（若系统提供）；
   - 不把冷启动首轮当作稳定性能。

### 7.2 M4 Pro 与 M5 Pro 的比较方法

不要直接比较两个不同时间点的单次结果。应：

1. 使用同一模型仓库、同一量化、同一 `mlx-vlm` 版本；
2. 使用同一 prompt、同一 tokenizer 版本、同一输出长度；
3. 两机都清空无关模型服务；
4. 各跑 3 次，报告 median；
5. 先比较 `prefill_tok/s` 和 `decode_tok/s`，再比较峰值内存与可用上下文；
6. 若结果差异小于运行波动，明确写“无显著差异”，不要过度归因芯片代际。

---

## 8. 执行顺序

```text
阶段 0：环境与资产核验
  ├─ 确认 3.5 SSH、磁盘、内存、后台进程
  ├─ 创建独立 mlx-vlm venv
  ├─ 锁定版本
  └─ 用已有 Qwen3.8-27B 做最小生成验证

阶段 1：Qwen3.8-27B 基线
  ├─ 128 / 1K / 4K / 8K / 16K
  ├─ 每场景预热 1 次 + 正式 3 次
  ├─ 记录 Prefill / Decode / 峰值内存 / swap
  └─ 阶梯探测最大可用上下文

阶段 2：Qwen3.6-35B-A3B
  ├─ ModelScope 断点下载约 19GiB
  ├─ 校验 4 个 safetensors 与配置
  ├─ 同矩阵评测
  └─ 单独分析 MoE 的 Decode 与内存局部性

阶段 3：Gemma-4-26B-A4B
  ├─ ModelScope 断点下载约 14.3GiB
  ├─ 校验 3 个 safetensors 与配置
  ├─ 同矩阵评测
  └─ 分析滑动窗口对长上下文的影响

阶段 4：M4 对照（可选）
  ├─ 在 3.7 建立相同 venv
  ├─ 停用/隔离 Ollama、oMLX、EXO 的大模型负载
  ├─ 只选 1K / 8K / 16K 三个代表场景
  └─ 与 3.5 做同条件对照

阶段 5：结果归档
  ├─ 原始 JSON/CSV
  ├─ 版本锁文件
  ├─ 机器状态快照
  └─ Markdown 报告
```

---

## 9. 风险与控制

| 风险 | 影响 | 控制措施 |
|---|---|---|
| 误用 `mlx-lm` 加载多模态模型 | 加载失败或结果不可比 | 统一使用 `mlx-vlm` |
| 三个模型量化口径不一致 | 速度/内存不可比 | 固定 `mlx-community` 4bit 仓库并记录 revision |
| 后台模型服务争抢 GPU/内存 | 数据抖动 | 测试前停止无关服务，记录进程快照 |
| Wi-Fi 下载不稳定 | 下载中断、耗时过长 | 优先有线；ModelScope + 断点续传 |
| 长上下文触发 swap | Decode 严重下降 | 记录 swap；把最大稳定上下文作为正式上限 |
| 只跑一次取最快值 | 结论不可复现 | 预热 + 3 次 median + 异常重跑 |
| 目录 `du` 被 LFS 缓存放大 | 内存预算误判 | 以 safetensors 实际字节数为准 |
| 共享机器被他人使用 | 结果受干扰 | 评测窗口预约；每轮前检查负载；异常组重跑 |

---

## 10. 当前实机状态与待办

### 10.1 已确认

- 3.5 是首选主测机；
- 3.13 不适合作为正式主测；
- 3.7 可作为 M4 对照机；
- 3.5 已有 `~/Models/Qwen3.8-27B-4bit`，有效权重约 14.95GiB；
- 3.5 已有 Homebrew `mlx-lm 0.32.0` / `mlx 0.32.3`；
- 3.5 原先没有 `mlx-vlm`；
- 三个目标模型都应使用 `mlx-vlm`；
- Qwen3.6-35B-A3B 与 Gemma-4-26B-A4B 尚未在 3.5 部署。

### 10.2 本轮环境准备动作

已在 3.5 创建：

```text
~/mlx-bench-venv
```

正在安装并验证：

```text
mlx-vlm 0.8.0
mlx 0.32.4
mlx-metal 0.32.4
transformers 5.24.0
huggingface-hub 2.4.0
opencv-python 5.1.0.95
```

已完成安装并生成：

```text
~/mlx-bench-venv/requirements-lock.txt
SHA256: 9f3d42d7f267cd5d14f09c94519f462391966d21ce320b58c82e53de21b3174d
```

`mlx_vlm.generate --help` 已验证可执行。新版 `mlx` 不暴露 `mlx.__version__`，版本应通过 `importlib.metadata.version("mlx")` 读取，不能把该属性缺失误判为安装失败。

### 10.3 已完成的最小加载验证

环境验证成功：

```text
mlx 0.32.4
mlx-metal 0.32.4
mlx-vlm 0.8.0
transformers 5.24.0
huggingface-hub 2.4.0
opencv-python 5.1.0.95
```

在 `192.168.3.5` 使用已有 `~/Models/Qwen3.8-27B-4bit` 执行短文本生成成功，日志：

```text
~/mlx-bench-venv/qwen38-smoke.log
~/mlx-bench-venv/qwen38-smoke-verbose.log
```

带 `--verbose` 的冒烟结果：

| prompt_tokens | 输出 tokens | Prefill | Decode | MLX peak memory | 进程最大 RSS | swap |
|---:|---:|---:|---:|---:|---:|---:|
| 28 | 64 | 约 62.1 tok/s | 约 18.3 tok/s | 约 16.4GB | 约 14.6GB（`maximum resident set size`） | 0 |

说明：

- 这是**单次冒烟，不是正式基准**；
- prompt 很短，Prefill 只代表极短序列；
- MLX 峰值内存与进程 RSS 的统计口径不同，不应混为同一个数值；
- 正式测试必须预热并重复 3 次后取 median。

### 10.4 下一步

1. ~~完成 `mlx-vlm` 安装并确认 CLI 可用~~（已完成）；
2. ~~用 Qwen3.8-27B 做最小文本生成~~（已完成冒烟）；
3. 下载 Qwen3.6-35B-A3B-4bit；
4. 下载 Gemma-4-26b-a4b-it-4bit；
5. 编写统一 benchmark 脚本；
6. 先在 3.5 跑完整矩阵；
7. 视需要再启用 3.7 对照；
8. 将原始数据与 Markdown 报告存入 `results/` 和 `docs/`。

---

## 11. 最终建议

**现在就从 `192.168.3.5` 开始，不要同时铺开三台机器。**

原因：

- 先建立单机、单引擎、单量化的可靠基线，才能判断后续差异来自模型、上下文还是机器；
- 3.5 已有 Qwen3.8 权重，能最快验证工具链；
- 3.13 的后台服务会污染结果；
- 3.7 适合作为第二阶段对照，而不是第一轮主测。

最终交付应至少包含：

1. 三个模型在 128 / 1K / 4K / 8K / 16K 下的 Prefill 与 Decode；
2. 峰值内存与 swap 曲线；
3. 每个模型的最大稳定上下文；
4. M4 Pro 与 M5 Pro 的同条件对照（如执行）；
5. 完整版本、模型 revision 和原始测量数据；
6. 一页结论：哪个模型在 64GB Mac 上更适合哪类上下文与负载。
