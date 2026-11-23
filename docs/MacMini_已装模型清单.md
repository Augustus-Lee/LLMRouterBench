# SynCo-MacMini（192.168.3.13）已装模型与部署方式

> 机器：`SynCo-MacMini`（Apple M5 Pro，18 核 CPU / 20 核 GPU，64GB 统一内存，926GB SSD）  
> IP：`192.168.3.13`，SSH alias：`macmini`，用户：`synco`  
> 最近复核：2026-11-23（Asia/Shanghai）

> **文档状态说明：** 本文已于 2026-11-23 按当前机器状态修订。三台 Mac mini 的统一评估与模型可行性判断请优先阅读 [`两台MacMini_配置与模型部署可行性评估.md`](./两台MacMini_配置与模型部署可行性评估.md)。该文件名保留历史称谓，内容已扩展为三台机器。

## 1. 当前机器概况

| 项目 | 当前值 |
|---|---|
| Hostname | `SynCo-MacMini` |
| 型号 | `Mac17,16` |
| 芯片 | Apple M5 Pro |
| CPU | 18 核（6 Super + 12 Performance） |
| GPU | 20 核 |
| 统一内存 | 64GB |
| 系统 | macOS 27.0（Build 26A425） |
| 数据卷 | 已用约 818GiB，可用约 63GiB，使用率 93% |
| Swap | 总量 2GB，当前已用约 443MB |
| 当前内存压力 | System-wide free 约 30%；Splash 正在常驻约 22.3GiB |

**底线判断：**

- 适合 Qwen3.8-27B、Qwen3.6-35B-A3B、Gemma-4-26B-A4B 的 4bit 级部署；
- 不适合把 Qwen3.8-Flash-125B-A6B 的 Q4 版本作为稳定基线；
- 63GiB 可用磁盘可以容纳 Gemma 的 15～16GB 权重，但不足以稳妥下载 125B 的 65～75GB Q4 权重；
- 正式评测前应释放不必要的磁盘缓存，并避免同时常驻多个大模型。

## 2. 四个目标模型：当前是否有

| 目标模型 | 是否已装 | 当前证据 | 结论 |
|---|---|---|---|
| **Qwen3.8-27B** | ✅ 已装 | MLX 4bit 约 15GB 有效权重；Splash GGUF Q4_K_M 约 15GB | 可直接评测 |
| **Qwen3.8-Flash-125B-A6B** | ❌ 未装 | 未发现权重或 Ollama 注册模型 | 不推荐 Q4 正式部署 |
| **Qwen3.6-35B-A3B** | ✅ 已装 | Ollama `qwen3.6:35b-a3b-q4_K_M`，约 23.9GB | 可直接评测 |
| **Gemma-4-26B-A4B** | ❌ 未装 | 未发现完整权重 | 硬件可运行，需安装约 15～16GB 4bit 模型 |

额外已有模型：

| 模型/目录 | 目录占用 | 说明 |
|---|---:|---|
| `~/Models/Qwen3.5-0.8B-4bit` | 约 1.2GB | MLX 小模型 |
| `~/Models/Qwen3.5-9B-4bit` | 约 11GB | MLX，目录含 LFS 副本 |
| `~/Models/Qwen3.5-35B-A3B-4bit` | 约 38GB | 有效权重约 19GB；版本是 **Qwen3.5**，不是目标 Qwen3.6 |

## 3. 模型目录与有效权重

`~/Models` 总计约 120GB，`~/.ollama` 约 22GB。

| 路径 | `du` 占用 | 有效内容 |
|---|---:|---|
| `~/Models/Qwen3.8-27B-4bit` | 约 30GB | 3 个 MLX safetensors 合计约 15GB；其余含 `.git/lfs` 副本 |
| `~/Models/splash/Qwen3.8-27B-GGUF` | 含在 splash 约 40GB 内 | Q4_K_M 约 15GB；mmproj 约 0.89GB；其他量化文件多为 LFS pointer |
| `~/Models/splash/Qwen3.8-27B-DFlash2` | 含在 splash 约 40GB 内 | draft model 权重约 3.6GB |
| `~/.ollama` | 约 22GB | 模型库包含 `qwen3.6:35b-a3b-q4_K_M` |

关键文件：

```text
Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf   约 15GB
Qwen3.8-27B-GGUF/mmproj-BF16.gguf             约 0.89GB
Qwen3.8-27B-DFlash2/model.safetensors          约 3.6GB
```

注意：很多同名 GGUF 量化文件只有 130～136 字节，是 Git LFS pointer，不是完整模型。评测脚本必须检查实际文件大小。

**版本登记提醒：** `Qwen3.8-27B-4bit/config.json` 的内部 `architectures` 为 `Qwen3_5ForConditionalGeneration`，`model_type` 为 `qwen3_5`。这可能是转换/打包命名差异，但正式报告必须记录原始 HF 仓库、revision 和转换流程。

## 4. 当前部署方式与服务拓扑

### 4.1 服务端口

| 端口 | 进程 | 用途 |
|---:|---|---|
| `127.0.0.1:18080` | Splash 1.5.0 | Qwen3.8-27B GGUF 的 OpenAI 兼容 API |
| `127.0.0.1:11435` | PAIR 内置 Ollama 0.33.3 | Qwen3.6-35B-A3B 原生 API |
| `*:8080` | NVIDIA PAIR | 聚合/代理入口；当前 model inventory 不可用 |
| `*:11434` | NVIDIA PAIR | Ollama 兼容代理 |
| `*:1234` | NVIDIA PAIR | LM Studio 兼容代理 |
| `*:14318`～`14323` | NVIDIA PAIR | 内部节点、引擎与调度服务 |

拓扑：

```text
Splash 127.0.0.1:18080  -> Qwen3.8-27B GGUF Q4_K_M
Ollama 127.0.0.1:11435  -> qwen3.6:35b-a3b-q4_K_M
NVIDIA PAIR 8080/11434/1234 -> 聚合与代理层
```

### 4.2 Splash

当前实际运行参数：

```text
模型：unsloth/Qwen3.8-27B-GGUF
权重：Qwen3.8-27B-UD-Q4_K_M.gguf
draft model：Qwen3.8-27B-DFlash2
mmproj：mmproj-BF16.gguf
host：127.0.0.1
port：18080
max-context：131072
max-memory：40G
kv-format：int8
default-reasoning-effort：medium
idle-release：10m
max-cache-disk：16G
queue-size：16
```

当前 `/v1/models` 返回：

```json
{
  "id": "qwen3.8-27b",
  "max_model_len": 131072,
  "context_length": 131072,
  "vision": true,
  "reasoning": {
    "thinks": true,
    "efforts": ["low", "medium", "xhigh"],
    "default": "medium"
  }
}
```

进程 RSS 约 22.3GiB。`--max-memory 40G` 是上限，不是实际常驻值。

`setup.sh` 内仍写 `0.0.0.0:8080`，但 8080 当前被 PAIR 占用；实际运行实例在 18080。直接执行旧脚本可能冲突，自动化时应先检查端口。

### 4.3 Ollama

当前真实 Ollama 在 `127.0.0.1:11435`：

```bash
curl -s http://127.0.0.1:11435/api/tags
```

模型元数据：

```text
name: qwen3.6:35b-a3b-q4_K_M
format: gguf
family: qwen35moe
parameters: 36.0B
quantization: Q4_K_M
size: 23,938,333,577 bytes
context_length: 262144
capabilities: vision, completion, tools, thinking
```

`262144` 是模型元数据支持值，不代表当前运行上下文已经设置为 262k。正式测试必须显式设置 `num_ctx`。

### 4.4 NVIDIA PAIR

PAIR 是聚合/路由层，不是模型权重本身。当前：

- `8080` 正在监听；
- `11434` 正在监听，但本次请求曾超时；
- `8080/v1/models` 返回 `model inventory unavailable`；
- 不能凭 PAIR 的模型列表推断权重已经存在；
- 若用于 LLMRouterBench 路由评测，必须记录底层实际模型和引擎。

## 5. 如何启动或调用

### 5.1 验证现有 Splash

```bash
ssh macmini
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
curl -s http://127.0.0.1:18080/v1/models | python3 -m json.tool
```

### 5.2 验证 Ollama

```bash
ssh macmini
curl -s http://127.0.0.1:11435/api/tags | python3 -m json.tool
```

调用示例：

```bash
curl -s http://127.0.0.1:11435/api/chat -d '{
  "model": "qwen3.6:35b-a3b-q4_K_M",
  "messages": [{"role": "user", "content": "你好"}],
  "stream": false
}'
```

### 5.3 启动 MLX-LM

旧脚本中的 `/Users/jiuwen` 应改为当前用户路径。建议先备份后修改：

```bash
sed -i '' 's|/Users/jiuwen|/Users/synco|g' ~/Models/Qwen3.8-27B-4bit/setup.sh
```

但该脚本使用 8080，与 PAIR 冲突。评测时建议复制命令并改用 8081：

```bash
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
mlx_lm.server \
  --model ~/Models/Qwen3.8-27B-4bit \
  --host 127.0.0.1 \
  --port 8081 \
  --max-tokens 4096 \
  --decode-concurrency 1 \
  --prompt-concurrency 1
```

### 5.4 安装 Gemma-4-26B-A4B

未安装。4bit 权重预计约 15～16GB，当前 63GiB 磁盘可以容纳，但应先确认下载源、revision 和可用磁盘。

```bash
df -h /System/Volumes/Data
```

建议先把 Splash 停止或等待空闲释放，再运行 Gemma，避免多模型常驻。不要在未确认仓库和文件大小前直接下载几十 GB 权重。

## 6. Qwen3.8-Flash-125B-A6B 的结论

### 6.1 为什么不适合

| 量化 | 权重估算 | 64GB 机器判断 |
|---|---:|---|
| Q4_K_M / 4bit | 约 65～75GB | ❌ 超过物理内存，不能稳定常驻 |
| Q3 / 3bit | 约 50～60GB | ⚠️ 很紧张，KV 和运行时易触发换页 |
| IQ2/Q2 | 约 35～45GB | ⚠️ 可能启动，但质量与长上下文能力下降 |

3.13 的额外限制：

1. 当前仅约 63GiB 可用磁盘，Q4 文件本身就不稳妥；
2. Splash 已常驻约 22.3GiB，不能再同时加载大模型；
3. 当前 swap 已用约 443MB，说明并非完全空载；
4. 64GB 内存还要留给 macOS、Metal、KV Cache 和请求峰值。

### 6.2 如果必须做实验

只能作为“低比特、短上下文、实验性”结果：

```text
量化：IQ2/Q2
上下文：4k～8k
并发：1
记录：加载时间、RSS、swap、失败率、Decode
标注：非正式 Q4 基线
```

MoE 的 A6B 只表示每 token 激活约 6B，不代表总权重可以无成本超过内存上限。非激活专家仍需要存储。

## 7. 当前性能观测与限制

`run.log` 近期记录：

| 指标 | 观测范围 |
|---|---|
| 输入 | 约 100～263 tokens |
| 输出 | 约 170～606 tokens |
| TTFT | 约 0.3～1.1 秒 |
| Decode | 约 28.7～49.5 tokens/s |
| 投机采样 | 约 3.5～5.6 tokens/cycle |

这些数字只能说明 Splash 链路工作正常，**不能直接作为正式 benchmark**。原因：

- 多个请求时间重叠，存在并发影响；
- 输入/输出长度不固定；
- thinking 状态与 reasoning effort 不固定；
- 采样参数未统一；
- 没有报告峰值 RSS、swap 和温度。

正式测试应固定：

```text
同一模型、同一量化、同一引擎
输入长度：1k / 8k / 32k / 64k
输出长度：256 / 512
并发：1，再测 2/4
温度：0 或固定值
预热：1 次
重复：3 次取中位数
指标：TTFT、Prefill tok/s、Decode tok/s、峰值 RSS、swap、失败率
```

## 8. 下一步

1. 用 Splash 在 18080 做 Qwen3.8-27B 的固定上下文基线；
2. 分别测试 DFlash2 开启/关闭与 INT8/BF16 KV Cache；
3. 释放 Splash 后，用 Ollama 测试 Qwen3.6-35B-A3B，固定 `num_ctx`；
4. 清理磁盘空间后安装 Gemma-4-26B-A4B 4bit，错峰测试；
5. 125B-A6B 不进入第一阶段正式评测；
6. 所有正式结果登记模型 revision、量化、引擎版本和上下文配置。
