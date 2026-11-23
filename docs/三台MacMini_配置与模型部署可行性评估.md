# 三台 Mac mini 配置核查与大模型部署可行性评估

> 核查时间：2026-11-23（Asia/Shanghai）  
> 核查对象：`192.168.3.5`（Jiuwen-MacMini）、`192.168.3.7`（Macmini-Admin）、`192.168.3.13`（SynCo-MacMini）  
> 核查方式：通过 SSH 远程读取 `system_profiler`、`sysctl`、`df`、`memory_pressure`、模型目录、运行时进程与本地 API。  
> 说明：本文只记录配置与判断，不记录登录密码、API Key、token 等敏感信息。

> 文档范围：本文答复“三台机器能否部署目标文本模型”，并为后续 Prefill、Decode、内存和上下文的 MAC 性能摸底提供基线。生图模型 Qwen-Image 与生视频模型 Wan2.2 暂不在本轮文本评测范围内。

## 1. 结论先行

### 1.1 三台机器的目标模型可用性

| 目标模型 | 192.168.3.5（M5 Pro / 64GB） | 192.168.3.7（M4 Pro / 64GB） | 192.168.3.13（M5 Pro / 64GB） | 结论 |
|---|---|---|---|---|
| Qwen3.8-27B | ✅ 已安装，可运行 | ⚠️ 未安装，硬件可运行 | ✅ 已安装 MLX 4bit 与 GGUF Q4_K_M | 两台 M5 可直接评测；M4 需复制或下载 |
| Qwen3.8-Flash-125B-A6B | ❌ 未安装；不推荐 Q4 | ❌ 未安装；不推荐 Q4 | ❌ 未安装；不推荐 Q4 | 三台 64GB 机器都不适合把 Q4 作为稳定基线 |
| Qwen3.6-35B-A3B | ✅ Ollama Q4_K_M 已安装 | ✅ Ollama Q4_K_M 已安装 | ✅ Ollama Q4_K_M 已安装 | 三台均可运行，适合作为 MoE 基线 |
| Gemma-4-26B-A4B | ⚠️ 未安装，硬件可运行 | ✅ MLX 4bit 已安装，另有 GGUF | ⚠️ 未安装，硬件可运行 | M4 直接可用；两台 M5 可复制后评测 |

### 1.2 最重要的判断

1. **三台机器都是 64GB 统一内存，适合 20B～35B 级模型的 4bit/8bit 本地推理。**
   - 它们不适合把 Qwen3.8-Flash-125B-A6B 的 Q4 版本作为常规稳定基线；
   - “模型文件接近 64GB”不等于“能稳定运行”，还必须为 macOS、Metal 缓冲区、KV Cache、运行时和请求峰值预留空间。

2. **Qwen3.8-Flash-125B-A6B 的瓶颈主要是内存，3.13 还额外受磁盘限制。**
   - Q4 权重通常约 65～75GB，超过 64GB 物理内存；
   - 3.13 当前只剩约 63GiB 可用磁盘，单是 Q4 文件就可能放不下；
   - 即使使用 mmap 或把权重放到 SSD，也会因换页导致加载慢、Decode 抖动和长上下文不稳定；
   - 若要摸底，只能使用 IQ2/Q2 级量化、4k～8k 短上下文，并明确标记为“实验性结果”。

3. **3.13（SynCo-MacMini）已经是一台可立即开展文本评测的机器。**
   - Qwen3.8-27B：Splash 正在提供 GGUF Q4_K_M 服务，当前暴露 131,072 tokens 上下文；
   - Qwen3.6-35B-A3B：Ollama 已安装 Q4_K_M，模型元数据报告 262,144 tokens 上下文；
   - Gemma-4-26B-A4B：未安装，但 64GB 内存与剩余磁盘在释放空间后可以运行 4bit 版本；
   - 125B-A6B：不建议纳入第一阶段正式评测。

4. **Splash、Ollama 与 NVIDIA PAIR 是不同层，评测时必须区分。**
   - Splash 是实际加载 Qwen3.8-27B GGUF 的推理服务；
   - Ollama 是实际加载 Qwen3.6-35B-A3B 的推理服务；
   - NVIDIA PAIR 是聚合/路由层，不能仅凭 PAIR 能列出模型就断定模型权重已安装；
   - 正式 benchmark 应直接访问底层引擎或至少记录 PAIR 实际路由到的后端。

5. **不能直接跨引擎比较速度。**
   - MLX-LM、Splash、Ollama、oMLX、LM Studio、llama.cpp 对 KV Cache、批处理、量化和 Metal 内核的实现不同；
   - 同一模型在不同引擎上的 TTFT/Decode 只能作为“引擎配置结果”，不能混成一个模型能力结论。

6. **三台机器都已经配置免密 SSH。**
   - 本机私钥为 `~/.ssh/id_ed25519`；
   - 公钥已安装到各机器的 `~/.ssh/authorized_keys`；
   - 已配置 `macmini`、`macmini-m5-jiuwen`、`macmini-m4-admin` 三个 alias。

## 2. SSH 免密访问配置

### 2.1 已完成的操作

使用本机现有 Ed25519 密钥对：

```text
私钥：~/.ssh/id_ed25519
公钥：~/.ssh/id_ed25519.pub
指纹：SHA256:KYXVCbbGdTs7PGvCpe+I5v2pOuGHnOKpS7UYm0dreQI
注释：augustus-codex-macmini
```

已将**公钥**追加到三台机器的：

```text
~/.ssh/authorized_keys
```

没有把私钥复制到远端。远程机器只需要公钥即可验证本机登录。

### 2.2 SSH alias

本机 `~/.ssh/config` 已配置：

```sshconfig
Host macmini
    HostName 192.168.3.13
    User synco
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new

Host macmini-m5-jiuwen
    HostName 192.168.3.5
    User jiuwen
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new

Host macmini-m4-admin
    HostName 192.168.3.7
    User admin
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
```

连接命令：

```bash
ssh macmini                 # 192.168.3.13 / synco
ssh macmini-m5-jiuwen       # 192.168.3.5  / jiuwen
ssh macmini-m4-admin        # 192.168.3.7  / admin
```

### 2.3 验证结果

```text
macmini           -> KEY_LOGIN_OK / SynCo-MacMini
macmini-m5-jiuwen -> KEY_LOGIN_OK / Jiuwen-MacMini
macmini-m4-admin  -> KEY_LOGIN_OK / Macmini-Admin
```

首次安装公钥时使用的登录密码没有写入 SSH 配置、仓库或本文档。

### 2.4 非交互 SSH 的 PATH 注意事项

在非交互 SSH 中，Homebrew 和 App 内置二进制不一定位于默认 `PATH`。自动化脚本建议显式设置：

```bash
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
```

或者使用完整路径，例如：

```text
/opt/homebrew/bin/splash
/opt/homebrew/opt/mlx-lm/bin/mlx_lm.generate
```

3.13 上的 Ollama 还是 NVIDIA PAIR 内置程序，系统 `PATH` 中通常没有 `ollama` 命令；这不代表 Ollama 未安装或未运行。

## 3. 三台机器的硬件与系统信息

### 3.1 192.168.3.5：Jiuwen-MacMini（M5 Pro / 64GB）

| 项目 | 实测值 |
|---|---|
| Hostname | `Jiuwen-MacMini` |
| 型号标识 | `Mac17,16` |
| 芯片 | Apple M5 Pro |
| CPU | 18 核：6 个 Super 核 + 12 个 Performance 核 |
| GPU | 20 核，Metal 4 |
| 统一内存 | 64GB |
| 系统 | macOS 27.2 |
| 内置磁盘 | 926GB |
| 数据卷已用 | 约 397GB |
| 当前可用空间 | 约 489GB |
| 内存压力 | 空闲时约 93% free，暂无异常压力 |

### 3.2 192.168.3.7：Macmini-Admin（M4 Pro / 64GB）

| 项目 | 实测值 |
|---|---|
| Hostname | `Macmini-Admin` |
| 型号标识 | `Mac16,11` |
| 芯片 | Apple M4 Pro |
| CPU | 14 核：10 个 Performance 核 + 4 个 Efficiency 核 |
| GPU | 20 核，Metal 4 |
| 统一内存 | 64GB |
| 系统 | macOS 26.6.2 |
| 内置磁盘 | 3.6TB |
| 数据卷已用 | 约 589GB |
| 当前可用空间 | 约 3.0TB |
| 内存压力 | 空闲时约 95% free，暂无异常压力 |

### 3.3 192.168.3.13：SynCo-MacMini（M5 Pro / 64GB）

| 项目 | 实测值 |
|---|---|
| Hostname | `SynCo-MacMini` |
| 型号标识 | `Mac17,16` |
| 芯片 | Apple M5 Pro |
| CPU | 18 核：6 个 Super 核 + 12 个 Performance 核 |
| GPU | 20 核 |
| 统一内存 | 64GB（68,719,476,736 bytes） |
| 系统 | macOS 27.0，Build 26A425 |
| 内置磁盘 | 926GB |
| 数据卷 `/System/Volumes/Data` | 已用约 818GiB，可用约 63GiB，使用率 93% |
| 当前内存状态 | System-wide free 约 30%；Splash 正在常驻约 22.3GiB |
| Swap | 总量 2048MB，已用约 443MB，空闲约 1605MB |

**3.13 的解释与限制：**

- 它和 3.5 同为 M5 Pro / 18 CPU / 20 GPU / 64GB，但两者系统版本和当前负载不同；
- 3.13 当前由 Splash 加载 Qwen3.8-27B，RSS 约 23,417,504KB，约 22.3GiB，约占物理内存 34.9%；
- 当前不是完全空闲机器，不能把此刻的 `memory_pressure` 当作空载基线；
- 63GiB 可用磁盘可以容纳 Gemma-4-26B-A4B 的 15～16GB 权重，但不适合再下载 125B 的 65～75GB Q4 权重；
- 正式测试长上下文前，应先清理无用缓存或外接存储，避免把磁盘换页混入性能结果。

### 3.4 硬件对推理性能的影响

本地大模型推理主要受以下因素影响：

1. **统一内存容量**
   - 决定权重、KV Cache、Metal 缓冲区和系统服务能否同时放下；
   - 64GB 机器不能只按“模型文件小于 64GB”判断可运行，还要为峰值预留空间。

2. **内存带宽**
   - Decode 阶段每生成一个 token 都要反复读取权重；
   - 内存带宽通常比 CPU 核心数更直接影响单请求 Decode；
   - M5 Pro 与 M4 Pro 的差异应以同模型、同量化、同引擎实测为准。

3. **GPU 核心数与 Metal 实现**
   - Prefill 和大矩阵计算主要受 GPU 核心、矩阵吞吐及内核实现影响；
   - 相同 GPU 核心数不代表相同性能，芯片代际、内存子系统和引擎版本都会改变结果。

4. **CPU 单核/多核**
   - 模型加载、tokenizer、采样、调度和 CPU fallback 会使用 CPU；
   - 长提示下 CPU 预处理和 tokenizer 也可能进入 TTFT；
   - 评测期间应关闭无关桌面应用和后台模型服务。

5. **上下文长度与 KV Cache**
   - Qwen3.8-27B 与 Qwen3.6-35B-A3B 的模型配置上限均为 262,144 tokens；
   - 3.13 的 Splash 实际只暴露 131,072 tokens，并使用 INT8 KV Cache；
   - 64GB 机器不应直接假设能跑满模型配置上限；
   - 建议至少测试 8k、32k、64k，长上下文再单独测试 128k，同时记录是否 OOM、换页或触发请求失败。

6. **闪存与模型加载**
   - 磁盘决定能否下载模型，也影响首次加载与 mmap 行为；
   - 权重完全驻留内存后，磁盘通常不是 Decode 主瓶颈；
   - 一旦发生 SSD 换页，速度会显著下降，且结果不再代表内存内推理。

7. **散热、电源与持续负载**
   - 持续 Prefill/Decode 可能引发温度和频率变化；
   - 应区分冷启动、预热后和持续 10 分钟三组结果；
   - “最短一次”不能作为稳定性能结论。

## 4. 192.168.3.13：已安装的模型与运行时

### 4.1 模型目录与磁盘占用

`~/Models` 总计约 120GB，`~/.ollama` 约 22GB。

| 模型/目录 | `du` 占用 | 有效内容 | 说明 |
|---|---:|---|---|
| `~/Models/Qwen3.8-27B-4bit` | 约 30GB | 3 个 safetensors 合计约 15GB | MLX 4bit；目录另含 `.git/lfs` 副本，不等于运行时常驻大小 |
| `~/Models/splash` | 约 40GB | GGUF 15GB + draft 3.6GB + mmproj 0.89GB | 目录含 LFS 副本与辅助文件 |
| `~/Models/Qwen3.5-35B-A3B-4bit` | 约 38GB | 约 19GB | MLX 4bit；版本是 **Qwen3.5**，不是目标 Qwen3.6 |
| `~/Models/Qwen3.5-9B-4bit` | 约 11GB | 约 10GB | 额外模型；目录含 LFS 副本 |
| `~/Models/Qwen3.5-0.8B-4bit` | 约 1.2GB | 约 1.2GB | 额外小模型 |
| `~/.ollama` | 约 22GB | `qwen3.6:35b-a3b-q4_K_M` 约 23.9GB 的逻辑模型大小 | Ollama 模型仓库；实际磁盘占用受共享 blob 影响 |

Splash 目录中的主要真实文件：

```text
Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf   约 15GB
Qwen3.8-27B-GGUF/mmproj-BF16.gguf             约 0.89GB
Qwen3.8-27B-DFlash2/model.safetensors          约 3.6GB
```

目录里的其他量化文件名多为 130～136 字节的 Git LFS pointer，不是完整权重；后续脚本必须检查文件大小，不能只看文件名。

**版本一致性提醒：**

- `Qwen3.8-27B-4bit/config.json` 的 `architectures` 为 `Qwen3_5ForConditionalGeneration`，`model_type` 为 `qwen3_5`；
- 这可能是上游转换/打包时的内部命名差异，但正式评测必须登记原始 Hugging Face 仓库、revision、文件名和转换脚本；
- 不应在论文或评测报告中只写展示名而忽略实际模型 revision。

### 4.2 当前服务与端口

| 端口 | 监听进程 | 当前用途 |
|---:|---|---|
| `127.0.0.1:18080` | Splash 1.5.0 | 直接提供 Qwen3.8-27B GGUF 的 OpenAI 兼容 API |
| `127.0.0.1:11435` | PAIR 内置 Ollama 0.33.3 | Qwen3.6-35B-A3B 的原生 Ollama API |
| `*:8080` | NVIDIA PAIR | 聚合/代理入口；当前模型 inventory 不可用 |
| `*:11434` | NVIDIA PAIR | 对外的 Ollama 兼容代理入口 |
| `*:1234` | NVIDIA PAIR | LM Studio 兼容代理入口 |
| `*:14318`～`14323` | NVIDIA PAIR 内部组件 | 节点、引擎、调度等内部端口 |

**当前实际拓扑应描述为：**

```text
Splash 127.0.0.1:18080  -> Qwen3.8-27B GGUF Q4_K_M
Ollama 127.0.0.1:11435  -> qwen3.6:35b-a3b-q4_K_M
NVIDIA PAIR 8080/11434/1234 -> 聚合与代理层
```

Splash 与 PAIR 当前可以并存，但 8080 已被 PAIR 占用。`~/Models/splash/Qwen3.8-27B-GGUF/setup.sh` 仍写着 `0.0.0.0:8080`，如果直接执行该脚本会与 PAIR 冲突；当前实际运行实例使用的是 `127.0.0.1:18080`。评测时以实际进程和 API 为准。

### 4.3 Splash 实际配置

当前进程的关键参数：

```text
服务：Splash 1.5.0
模型：unsloth/Qwen3.8-27B-GGUF
权重：Qwen3.8-27B-UD-Q4_K_M.gguf
draft model：Qwen3.8-27B-DFlash2
mmproj：mmproj-BF16.gguf
地址：127.0.0.1
端口：18080
最大上下文：131072
最大内存参数：40G
KV Cache：INT8
默认推理强度：medium
空闲释放：10 分钟
磁盘缓存：16G
队列：16
```

进程实际 RSS 约 22.3GiB，低于 `--max-memory 40G`。40G 是上限，不是当前常驻量；论文或报告中不能把 40G 写成“实际占用 40G”。

`/v1/models` 当前返回：

```json
{
  "id": "qwen3.8-27b",
  "max_model_len": 131072,
  "context_length": 131072,
  "vision": true,
  "input_modalities": ["text", "image", "pdf"],
  "reasoning": {
    "thinks": true,
    "efforts": ["low", "medium", "xhigh"],
    "default": "medium"
  }
}
```

DFlash2 的 draft 配置包含 `block_size: 8`、5 层 draft 模型和 2048 的滑动窗口。它用于投机采样加速 Decode，会改变“纯逐 token 解码”的基线；正式测试应分别记录：

- DFlash2 开启；
- DFlash2 关闭；
- INT8 KV Cache；
- BF16 KV Cache（若显存允许）；
- 不同 reasoning effort。

### 4.4 运行时观测数据

`run.log` 中近期请求表现为：

| 指标 | 观测范围 |
|---|---|
| 输入 tokens | 约 100～263 |
| 输出 tokens | 约 170～606 |
| TTFT | 约 0.3～1.1 秒 |
| Decode | 约 28.7～49.5 tokens/s |
| 投机采样 | 约 3.5～5.6 tokens/cycle |

**这些数据只用于判断服务已正常工作，不能直接作为正式 benchmark。** 日志中多个请求时间重叠，且输入长度、输出长度、thinking 状态和并发不固定；正式评测必须控制提示长度、输出长度、并发数、预热与采样参数，并单独测量 Prefill 和 Decode。

### 4.5 Ollama / NVIDIA PAIR

Ollama API：

```bash
curl -s http://127.0.0.1:11435/api/tags
```

当前模型：

```text
qwen3.6:35b-a3b-q4_K_M
  format: gguf
  family: qwen35moe
  parameters: 36.0B
  quantization: Q4_K_M
  size: 23,938,333,577 bytes（约 23.9GB）
  metadata context_length: 262144
  capabilities: vision, completion, tools, thinking
```

注意：

- `11435` 是实际 Ollama；
- `11434` 是 NVIDIA PAIR 的代理入口，本次请求曾出现超时；
- `8080/v1/models` 当前返回 `{"error":"model inventory unavailable"}`；
- `262144` 是模型元数据支持值，不等于当前运行实例已经启用 262k；
- 正式测试必须显式设置 `num_ctx`，并记录模型是否被同时加载。

### 4.6 3.13 的关键风险

1. **磁盘余量偏低**：仅约 63GiB，下载和缓存会快速压缩余量；
2. **已有常驻模型**：Splash 当前 RSS 约 22.3GiB，Ollama 若同时加载 35B 会显著提高总占用；
3. **Swap 已使用**：当前约 443MB，说明机器并非完全轻载；正式跑分前应建立空载基线；
4. **版本命名需固定**：`Qwen3.8-27B` 内部配置显示 `qwen3_5`，必须核对来源；
5. **端口配置不一致**：setup 脚本写 8080，实际服务在 18080，自动化脚本不能盲目复用旧命令；
6. **PAIR 不是模型证据**：判断模型是否可用必须看底层权重、进程和 API。

## 5. 192.168.3.5：已安装的模型与运行时

### 5.1 模型目录

| 模型/目录 | 占用 | 说明 |
|---|---:|---|
| `~/Models/Qwen3.8-27B-4bit` | 目录约 30GB；有效权重约 15GB | MLX safetensors；`.git/lfs` 另占约 15GB，不是运行时常驻内存 |
| `~/Models/splash/Qwen3.8-27B-GGUF` | 目录约 32GB；有效 Q4 约 15GB | 含辅助文件和 `.git/lfs` 副本；实际 Q4_K_M 约 15GB |
| `~/Models/splash/Qwen3.8-27B-DFlash2` | 目录约 7.2GB；有效权重约 3.6GB | Splash 的 draft model |
| `~/Models/Qwen3.5-35B-A3B-4bit` | 目录约 38GB；有效权重约 19GB | Qwen3.5，不是目标 Qwen3.6 |
| `~/Models/Qwen3.5-9B-4bit` | 目录约 11GB；有效权重约 10GB | 额外模型 |
| `~/Models/Qwen3.5-0.8B-4bit` | 约 1.2GB | 额外小模型 |
| `~/.ollama` | 约 22GB | 含 `qwen3.6:35b-a3b-q4_K_M` |

### 5.2 已安装的运行时

| 运行时 | 状态 | 端口 | 用途 |
|---|---|---|---|
| MLX-LM | `mlx-lm 0.32.0`，可用 | 默认可配 8080/8081 | Apple MLX 原生推理 |
| Splash | `1.5.0`，已安装 | setup 写 8080 | GGUF + Metal + DFlash2 |
| Ollama | `0.33.3`，PAIR 内置 | 11435 | GGUF 模型服务 |
| LM Studio | 已运行 `llmster` | 1234/1235 | 模型管理与本地 API |
| NVIDIA PAIR | 已运行 | 8080、11434、1234 等 | 聚合/路由网关 |

### 5.3 当前 API 快照

```text
http://127.0.0.1:11434/api/tags
  -> 代理入口，本次曾超时

http://127.0.0.1:11435/api/tags
  -> qwen3.6:35b-a3b-q4_K_M

http://127.0.0.1:1234/v1/models
  -> qwen3.5-35b-a3b
  -> text-embedding-nomic-embed-text-v1.5

http://127.0.0.1:8080/v1/models
  -> 当前返回 model inventory unavailable
```

说明：3.5 当前没有 Splash 进程监听。若要启动 Splash，应先解决 8080 与 PAIR 的端口冲突，并固定上下文、KV 类型和 draft 设置。

## 6. 192.168.3.7：已安装的模型与运行时

### 6.1 模型与模型文件

| 模型/目录 | 占用 | 说明 |
|---|---:|---|
| `~/.omlx/models/gemma-4-26b-a4b-it-4bit` | 约 15GB | MLX 4bit，Gemma-4-26B-A4B |
| `~/.omlx/models/Qwen3.5-9B-MLX-8bit` | 约 9.7GB | 额外模型 |
| `~/.ollama/models` | 约 28GB | Ollama 模型仓库 |
| `~/.lmstudio/models/lmstudio-community/Qwen3.5-35B-A3B-GGUF` | 约 34GB | LM Studio 的 Q8_0 GGUF |
| `~/workspace/GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` | 约 16GB | 可直接用于 llama.cpp 或其他 GGUF 引擎 |
| `~/.exo/models` | 约 6.7GB | 主要是 EXO 元数据和缓存，不是完整大模型仓库 |

Ollama 中实际存在：

```text
qwen3.6:35b-a3b-q4_K_M
  参数量：36.0B
  量化：Q4_K_M
  模型大小：约 22GB（API 逻辑大小约 23.9GB）
  上下文：262144
  能力：vision、completion、tools、thinking

qwen3.5:9b
  量化：Q4_K_M
  模型大小：约 6.1GB
  上下文：262144
```

### 6.2 已安装的运行时

| 运行时 | 状态 | 端口 | 说明 |
|---|---|---|---|
| oMLX | 已安装，当前服务未监听 | 配置为 8000 | MLX 模型管理与 OpenAI 兼容服务 |
| Ollama | 已运行 | 11434 | 主要 GGUF 推理服务 |
| LM Studio | 已运行 | 1234 | 本地模型服务 |
| EXO | 已运行 | 52415 | 分布式/多设备推理框架 |
| NVIDIA PAIR | 已运行 | 8080 等 | 聚合/路由 |
| llama.cpp | 源码在工作区 | 未发现 PATH 中的 `llama-server` | 可从 `~/workspace/my_ollama/llama.cpp` 构建 |

### 6.3 oMLX 当前配置要点

```text
模型目录：/Users/admin/.omlx/models
最大上下文窗口：32768
最大输出 tokens：32768
最大模型内存：50.40GB
API 认证：已启用
默认模型：gemma-4-26b-a4b-it-4bit
```

注意：oMLX 配置中存在明文 API key 字段。本文不复制该值；后续应改为环境变量或重新生成并轮换，不要把配置文件提交到 Git。

### 6.4 EXO 的模型目录说明

EXO API 可以列出大量模型名称，但 `~/.exo/models` 中大部分目录只有元数据，不代表完整权重已经下载。判断“已安装”必须以实际大文件（safetensors/GGUF）和可启动 API 为准。

## 7. 四个目标模型的逐项可行性

### 7.1 Qwen3.8-27B

**现状**

- 192.168.3.5：已安装 MLX 4bit 与 GGUF Q4_K_M；
- 192.168.3.7：未发现完整模型目录；
- 192.168.3.13：已安装 MLX 4bit 与 GGUF Q4_K_M，且 Splash 正在运行。

**部署判断**

| 机器 | 判断 | 建议 |
|---|---|---|
| 192.168.3.5 M5 Pro 64GB | ✅ 推荐 | MLX-LM 作为主基线；Splash 做投机采样对照 |
| 192.168.3.7 M4 Pro 64GB | ⚠️ 可行但未安装 | 复制或下载 4bit MLX / Q4_K_M GGUF 后运行 |
| 192.168.3.13 M5 Pro 64GB | ✅ 推荐 | 复用 Splash 的 18080 服务；另启 MLX 做引擎对照 |

**内存与上下文**

- 有效权重约 15GB；Splash 还加载约 3.6GB draft model 和约 0.89GB mmproj；
- 3.13 当前 Splash RSS 约 22.3GiB，服务参数上限为 40GB；
- 模型配置支持 262,144 tokens，但 3.13 当前服务只暴露 131,072 tokens；
- 建议先测 8k、32k、64k，再单独测试 128k；
- 必须在结果中记录 KV Cache 是 INT8 还是 BF16，以及是否开启 DFlash2。

### 7.2 Qwen3.8-Flash-125B-A6B

**现状**

- 三台机器均未发现完整权重；
- 未发现 125B/A6B 对应的大文件或已注册 Ollama 模型；
- 3.5 与 3.7 的磁盘足以存放 Q4 文件，但这不能解决 64GB 内存问题；
- 3.13 当前仅约 63GiB 可用磁盘，连 Q4 文件空间都不稳妥。

**容量估算**

| 量化级别 | 权重大致体积 | 64GB 机器判断 |
|---|---:|---|
| Q4_K_M / 4bit | 约 65～75GB | ❌ 超过总内存，不能稳定常驻 |
| Q3 / 3bit | 约 50～60GB | ⚠️ 很紧张，KV 和运行时容易触发换页 |
| IQ2/Q2 | 约 35～45GB | ⚠️ 可能启动，但质量和长上下文能力明显下降 |

**结论**

- **不建议把 125B-A6B Q4 作为三台 64GB Mac mini 的正式评测模型。**
- 若必须摸底，建议：
  1. 仅使用 IQ2/Q2 级量化；
  2. 上下文限制在 4k～8k；
  3. 记录 swap、内存压力、加载时间、请求失败率和有效 Decode；
  4. 结果标注为“低比特实验”，不与 Q4 结果直接比较；
  5. 如要做 Q4 级实验，优先考虑多机分布式推理，但这会引入网络通信和分片开销。
- MoE 的“A6B”只表示每 token 激活参数较少，**不代表总权重可以超过 64GB 而不付出代价**；非激活专家仍需要存储。

### 7.3 Qwen3.6-35B-A3B

**现状**

- 192.168.3.5：Ollama 已安装 `qwen3.6:35b-a3b-q4_K_M`，实际 Ollama 监听 11435；
- 192.168.3.7：Ollama 已安装同一模型，监听 11434；
- 192.168.3.13：Ollama 已安装同一模型，监听 11435；
- 3.5 与 3.13 另有 `Qwen3.5-35B-A3B-4bit`，但版本是 3.5，不应冒充 3.6。

**部署判断**

| 机器 | 判断 | 建议 |
|---|---|---|
| 192.168.3.5 M5 Pro 64GB | ✅ 推荐 | Ollama Q4_K_M；可再测 MLX 版本 |
| 192.168.3.7 M4 Pro 64GB | ✅ 推荐 | Ollama Q4_K_M；LM Studio Q8_0 可做精度/速度对照 |
| 192.168.3.13 M5 Pro 64GB | ✅ 推荐 | Ollama Q4_K_M；评测时不要与 Splash 的 27B 同时常驻 |

**内存与上下文**

- Q4_K_M 报告约 23.9GB，64GB 机器有运行余量；
- 与 Splash 的 Qwen3.8-27B 同时常驻时，总模型内存可能接近或超过 50GB，再叠加 KV 和系统缓冲区会变得紧张；
- 8bit 约 34～36GB 可以尝试，但应减少并发；
- bf16 约 67GB，不适合 64GB 机器稳定部署；
- 模型元数据支持 262,144 tokens，但实际上下文必须以 `num_ctx` 和运行结果为准。

### 7.4 Gemma-4-26B-A4B

**现状**

- 192.168.3.5：未发现完整模型，但 64GB 内存足够运行 4bit；
- 192.168.3.7：已安装 MLX 4bit（约 15GB）和 16GB GGUF；
- 192.168.3.13：未发现完整模型；63GiB 磁盘可容纳 4bit，但需先确认是否清理空间。

**部署判断**

| 机器 | 判断 | 建议 |
|---|---|---|
| 192.168.3.5 M5 Pro 64GB | ⚠️ 可运行但未安装 | 从 M4 复制 15GB MLX 或 16GB GGUF |
| 192.168.3.7 M4 Pro 64GB | ✅ 推荐 | oMLX 或 llama.cpp/MLX-LM 运行 4bit |
| 192.168.3.13 M5 Pro 64GB | ⚠️ 可运行但未安装 | 安装后单独评测，先释放 Splash 常驻内存 |

**内存与上下文**

- 4bit 约 15GB，内存余量充足；
- 6bit 约 21GB、8bit 约 27GB 也可尝试；
- bf16 约 49GB，属于“能加载但长上下文余量很小”；
- oMLX 当前只启用 32768 上下文，不能把模型配置中的 262144 当作已启用值。

## 8. 本地部署方式详解

### 8.1 MLX-LM

**原理**

- Apple MLX 框架上的 Python 推理栈；
- 直接加载 MLX 格式 safetensors；
- 使用 Apple Silicon 的 Metal GPU 和统一内存；
- 提供 `mlx_lm.generate` 命令行和 `mlx_lm.server` OpenAI 兼容 HTTP 服务。

**示例**

```bash
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# 交互式生成
mlx_lm.generate --model ~/Models/Qwen3.8-27B-4bit --prompt "你好"

# 启动本地服务
mlx_lm.server \
  --model ~/Models/Qwen3.8-27B-4bit \
  --host 127.0.0.1 \
  --port 8081 \
  --max-tokens 4096 \
  --decode-concurrency 1 \
  --prompt-concurrency 1
```

**优点**

- 与 Apple Silicon 结合紧密；
- 4bit/8bit MLX 生态成熟；
- 适合作为 Prefill/Decode 的基础性能基线。

**注意事项**

- 不能直接加载 GGUF；
- 服务版本、并发、prompt cache 和采样参数会显著影响 benchmark；
- 3.5/3.13 的旧 `setup.sh` 可能包含 `/Users/jiuwen` 路径，直接执行前应检查并修正。

### 8.2 Splash

**原理**

- 面向本地 GGUF 的推理服务；
- 使用 Metal 加速；
- 支持 draft model 投机采样和 mmproj 多模态投影；
- 提供 OpenAI 兼容接口。

**3.13 当前配置**

- Splash 1.5.0；
- Qwen3.8-27B GGUF Q4_K_M；
- DFlash2 投机采样；
- INT8 KV Cache；
- 131,072 tokens 上下文；
- 127.0.0.1:18080；
- 最大内存参数 40G，当前 RSS 约 22.3GiB。

**启动前检查**

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
lsof -nP -iTCP:18080 -sTCP:LISTEN
```

由于 3.13 的 8080 已被 NVIDIA PAIR 占用，直接执行旧的 `setup.sh` 可能失败或改变现有服务。评测脚本应优先复用已有 18080 服务，或复制脚本后显式修改端口与运行参数。

### 8.3 Ollama

**原理**

- 本地模型管理和推理服务；
- 模型通常为 GGUF；
- 默认 API 为 `http://127.0.0.1:11434`；
- 支持 thinking、tools、vision 等能力。

**三台机器上的情况**

- 3.5：PAIR 内置 Ollama 0.33.3，实际监听 11435；
- 3.7：Ollama.app 监听 11434；
- 3.13：PAIR 内置 Ollama 0.33.3，实际监听 11435；
- 三台均有 `qwen3.6:35b-a3b-q4_K_M`。

**查询模型**

```bash
curl -s http://127.0.0.1:11434/api/tags | python3 -m json.tool
curl -s http://127.0.0.1:11435/api/tags | python3 -m json.tool
```

**注意事项**

- 11434 与 11435 在不同机器上角色不同，不能用一套 URL 假设所有机器；
- 默认参数不一定是公平 benchmark 参数；
- 需要确认实际量化、上下文、GPU/CPU 分层和并发；
- 模型元数据中的 `context_length` 不是当前运行上下文。

### 8.4 oMLX

**原理**

- M4 上安装的 Apple Silicon 本地模型服务；
- 主要使用 MLX 权重；
- 提供模型发现、固定、KV/SSD 缓存和 OpenAI 兼容 API；
- 当前模型目录为 `~/.omlx/models`。

**M4 上的现状**

- Gemma-4-26B-A4B 4bit 已加载过；
- 日志显示模型估算占用约 15.26GB；
- 运行时最大上下文配置为 32768；
- 服务当前没有监听 8000。

**注意事项**

- API 认证已开启，调用需要对应 Bearer token；
- token 不应写入 Git；
- 配置文件当前有明文 key，建议轮换并改用安全存储。

### 8.5 LM Studio

**原理**

- 图形化模型管理与本地推理服务；
- 可加载 GGUF/MLX 模型；
- 提供 OpenAI 兼容 API，M4 上监听 1234。

**现状**

- 3.5：可列出 `qwen3.5-35b-a3b` 和 embedding；
- 3.7：可列出 `qwen3.5-35b-a3b` 和 embedding；
- 3.7 的 `Qwen3.5-35B-A3B-Q8_0.gguf` 约 34GB。

**注意事项**

- GUI 默认上下文、GPU offload 和采样设置会影响结果；
- 正式 benchmark 时应固定配置并记录版本；
- 模型展示名必须与底层文件版本对应。

### 8.6 EXO

**原理**

- 面向多设备的分布式推理框架；
- 可将模型分片到多台 Apple Silicon 机器；
- M4 当前监听 52415。

**现状**

- M4 已安装 EXO.app，API 可访问；
- catalog 中大量模型名不代表本地已有完整权重；
- 尚未验证三台机器是否都加入同一 EXO 集群。

**适用场景**

- 未来尝试 125B 级模型的多机分布式实验；
- 对比单机与集群的吞吐/延迟；
- 需要额外处理网络、分片、兼容性和故障恢复。

### 8.7 llama.cpp

**原理**

- C/C++ 实现的 GGUF 推理引擎；
- 支持 CPU、Metal、mmap 和多种量化；
- 适合做跨引擎基线与低比特实验。

**M4 现状**

- 工作区有 `~/workspace/my_ollama/llama.cpp` 源码；
- PATH 中没有可直接执行的 `llama-server`；
- 可用已有 Gemma GGUF 构建后测试。

```bash
cd ~/workspace/my_ollama/llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j
./build/bin/llama-server \
  -m ~/workspace/GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8082 \
  -c 32768 -ngl 99
```

具体编译参数以仓库当前版本为准。

### 8.8 NVIDIA PAIR

**原理**

- 本地多引擎的聚合与路由网关；
- 3.5 与 3.13 都在多端口运行；
- 适合统一 OpenAI 接口和多模型路由。

**注意**

- PAIR 不是“模型已下载”的证据；
- 3.13 的 8080 被 PAIR 占用，与 Splash 默认端口冲突；
- 若用于 LLMRouterBench 路由评测，必须记录 PAIR 实际选择的后端、底层模型和推理引擎；
- 当前 3.13 的 PAIR model inventory 返回错误，不能直接作为模型可用性证明。

## 9. 推荐评测落地方案

### 9.1 第一阶段：文本模型基线

固定以下变量：

```text
量化：优先 4bit / Q4_K_M
引擎：MLX-LM、Splash、Ollama 分开统计
上下文：8k、32k、64k
输出长度：固定 256/512 tokens
并发：先单请求，再测 2/4 并发
温度：0 或固定贪心
预热：每个配置预热 1 次，正式跑 3 次取中位数
指标：TTFT、Prefill tok/s、Decode tok/s、峰值内存、swap、加载时间、失败率
```

### 9.2 模型与机器分配

| 机器 | 主力模型 | 用途 |
|---|---|---|
| 192.168.3.5 M5 Pro 64GB | Qwen3.8-27B、Qwen3.6-35B-A3B | MLX/GGUF 与 MoE 基线 |
| 192.168.3.7 M4 Pro 64GB | Gemma-4-26B-A4B、Qwen3.6-35B-A3B | M4 硬件基线与多引擎对照 |
| 192.168.3.13 M5 Pro 64GB | Qwen3.8-27B、Qwen3.6-35B-A3B，安装 Gemma 后加入 | 现成 Splash 服务；重点做引擎和上下文对照 |
| 三台联合 | 125B-A6B（仅实验） | 仅在低比特、短上下文或 EXO 分布式方案下尝试 |

### 9.3 3.13 的建议执行顺序

1. **先测 Qwen3.8-27B 的 Splash 基线**
   - 固定 8k、32k、64k；
   - 记录 TTFT、Decode、tokens/cycle、RSS、swap；
   - 再分别做 DFlash2 开启/关闭对照。

2. **停止或释放 Splash 后测 Qwen3.6-35B-A3B**
   - 通过 11435 调用 Ollama；
   - 固定 `num_ctx`；
   - 不应把 Splash 的 27B 常驻内存计入 35B 的“模型内存”。

3. **安装 Gemma-4-26B-A4B 4bit**
   - 磁盘可容纳约 15～16GB；
   - 建议与 Splash 错峰运行；
   - 统一使用 MLX 或 llama.cpp 后与其他机器比较。

4. **125B-A6B 暂不进入正式基线**
   - 如必须做实验，使用 IQ2/Q2、4k～8k 上下文；
   - 需先解决 3.13 磁盘空间问题；
   - 结果单独成表并标注实验条件。

### 9.4 125B 的建议

不建议把 125B-A6B Q4 作为第一轮评测对象。更稳妥的顺序是：

1. 先完成 27B、35B-A3B、Gemma-4-26B-A4B 的 Q4 基线；
2. 确认单机内存峰值、长上下文表现和引擎差异；
3. 再下载 125B 的 IQ2/Q2 版本，做 4k/8k 短上下文实验；
4. 若必须测 Q4，优先考虑多机分布式推理，并单独报告网络开销和失败率；
5. 所有低比特结果都标注量化等级，不与 Q4 结果直接横向比较。

## 10. 目前可直接执行的核查命令

```bash
# 3.13 SynCo-MacMini
ssh macmini
hostname
sw_vers
sysctl -n hw.model
sysctl -n machdep.cpu.brand_string
sysctl -n hw.ncpu
sysctl -n hw.memsize
df -h /System/Volumes/Data
memory_pressure
sysctl vm.swapusage
curl -s http://127.0.0.1:18080/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:11435/api/tags | python3 -m json.tool
lsof -nP -iTCP -sTCP:LISTEN | grep -E ':(8080|11434|11435|1234|18080) '

# 3.5 Jiuwen-MacMini
ssh macmini-m5-jiuwen
hostname
df -h /System/Volumes/Data
memory_pressure
curl -s http://127.0.0.1:11435/api/tags | python3 -m json.tool

# 3.7 Macmini-Admin
ssh macmini-m4-admin
hostname
df -h /System/Volumes/Data
memory_pressure
curl -s http://127.0.0.1:11434/api/tags | python3 -m json.tool
ls -lh ~/.omlx/models/gemma-4-26b-a4b-it-4bit
ls -lh ~/workspace/GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
```

## 11. 最终建议

1. **立即开始评测的模型：**
   - 3.5：Qwen3.8-27B（MLX 4bit）、Qwen3.6-35B-A3B（Ollama Q4_K_M）；
   - 3.7：Gemma-4-26B-A4B（MLX 4bit）、Qwen3.6-35B-A3B（Ollama Q4_K_M）；
   - 3.13：Qwen3.8-27B（Splash GGUF Q4_K_M）、Qwen3.6-35B-A3B（Ollama Q4_K_M）。

2. **需要先准备的模型：**
   - 两台 M5 安装 Gemma-4-26B-A4B 4bit；
   - M4 安装 Qwen3.8-27B 4bit（若要做同模型跨机对比）；
   - 3.13 在安装 Gemma 前先释放磁盘空间并确认 Splash 是否停止。

3. **暂时不要作为正式基线：**
   - Qwen3.8-Flash-125B-A6B Q4；
   - 任何超过 64GB 内存预算且需要长上下文的模型；
   - 未记录实际后端和量化版本的 PAIR 路由结果。

4. **正式评测必须记录：**
   - 模型原始仓库与 revision；
   - 量化格式和实际文件大小；
   - 推理引擎及版本；
   - 上下文长度、KV Cache 类型；
   - 并发数、预热次数、采样参数；
   - 峰值内存、swap、温度和持续运行时间；
   - 是否启用 speculative decoding、vision projection 或其他加速。

---

**一句话总结：** 三台 64GB Mac mini 都能较好地运行 Qwen3.8-27B、Qwen3.6-35B-A3B 和 Gemma-4-26B-A4B 的 4bit 版本；其中 3.13 的 Qwen3.8-27B 已由 Splash 以 128K 上下文运行、Qwen3.6-35B-A3B 已由 Ollama 安装，但磁盘只剩约 63GiB 且不适合 125B Q4。Qwen3.8-Flash-125B-A6B 受 64GB 内存限制，只适合低比特、短上下文或多机分布式实验，不应作为正式 Q4 基线。
