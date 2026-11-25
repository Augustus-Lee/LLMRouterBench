# Mac 算力、带宽与机器资源实时监控接入

> 适用范围：Apple Silicon Mac mini 上的文本大模型性能评测  
> 目标：在不把推导值冒充硬件计数器的前提下，把 CPU、统一内存、GPU/NPU 占用、功耗、温控、token 吞吐、有效算力和有效带宽接入实时监控  
> 关联脚本：`tools/monitor_machine.py`、`tools/bench_accuracy_concurrency.py`  
> 最近更新：2026-11-25

## 1. 先明确指标口径

Apple Silicon 不能通过普通 `powermetrics` 直接读取“实时物理 TFLOPS 计数器”或“GPU/NPU 物理总线带宽计数器”。因此本方案把指标分成三层，报告、图表和 JSON 字段都必须保持同一口径。

### 1.1 L1：硬件实时状态

由 `powermetrics` 和系统命令采集，表示设备当前运行状态：

- CPU 使用率、每核负载；
- 统一内存 free/active/inactive/wired/compressor、swap；
- GPU active residency、GPU HW active frequency、GPU power；
- NPU/ANE active residency、HW active frequency、ANE power；
- thermal pressure；
- SSD 读写速率；
- 推理进程 RSS（由 bench 的 `SystemMonitor` 采样）。

这些字段能回答“GPU 忙不忙、频率多少、功耗多少、是否 swap、是否热降频”，但不能直接等价为“模型实际完成了多少 TFLOPS”或“GPU 总线实际传输了多少 GB/s”。

### 1.2 L2：workload-level 有效值

由**已完成请求的 token 吞吐**和显式提供的每 token FLOPs/字节数推导：

```text
input_tokens_per_s  = window 内 prompt_tokens 之和 / window_s
output_tokens_per_s = window 内 completion_tokens 之和 / window_s

prefill_effective_tflops =
    input_tokens_per_s × prefill_flops_per_token / 1e12

decode_effective_tflops =
    output_tokens_per_s × decode_flops_per_token / 1e12

prefill_effective_bandwidth_gb_s =
    input_tokens_per_s × prefill_bytes_per_token / 1e9

decode_effective_bandwidth_gb_s =
    output_tokens_per_s × decode_bytes_per_token / 1e9
```

其中：

- `prefill_effective_tflops` 和 `decode_effective_tflops` 是分阶段有效吞吐；
- `effective_tflops` 是两者之和，属于聚合推导值；
- `prefill_effective_bandwidth_gb_s` / `decode_effective_bandwidth_gb_s` 是分阶段有效字节吞吐；
- `effective_bandwidth_gb_s` 是两者之和，属于聚合推导值；
- 所有公式中的 `<...>_per_token` 必须由本次模型、量化、实现和激活路径推导，不能只根据总参数猜测。

### 1.3 L3：理论基线

`--theoretical-tflops` 和 `--official-bandwidth-gb-s` 只用于计算利用率分母：

```text
tflops_utilization = effective_tflops / theoretical_tflops
bandwidth_utilization = effective_bandwidth_gb_s / official_bandwidth_gb_s
```

理论值不能写成“实时实测值”。没有可靠来源时保持 `null`，不要为了图表完整而填写估算值。

### 1.4 三类值不得混写

| 类型 | 典型字段 | 是否来自硬件计数器 | 正确表述 |
|---|---|---|---|
| 硬件实时状态 | `gpu.gpu_usage_percent`、`gpu.gpu_freq_mhz`、`gpu.gpu_power_mw` | 否，来自 `powermetrics` 采样 | GPU 占用率 / active frequency / 功耗 |
| workload 有效值 | `compute.prefill_effective_tflops`、`compute.decode_effective_bandwidth_gb_s` | 否，由 token 吞吐推导 | Prefill/Decode 有效算力 / 有效带宽 |
| 理论基线 | `theoretical_tflops`、`official_bandwidth_gb_s` | 否，来自硬件规格或推导文档 | 理论峰值 / 利用率分母 |

## 2. 启动 `powermetrics`

`powermetrics` 通常需要管理员权限，建议在独立终端或 tmux 窗口运行。使用绝对路径，避免 bench 和 monitor 的工作目录不同导致找不到文件：

```bash
sudo powermetrics \
  --samplers cpu_power,gpu_power,ane_power,thermal \
  -i 2000 \
  -o /Users/jiuwen/powermetrics_qwen38.txt
```

参数说明：

- `-i 2000`：约每 2 秒采样一次；
- `-o`：持续追加到文本文件；
- `gpu_power`：GPU active residency、frequency、power 等字段；
- `ane_power`：NPU/ANE 相关字段；
- `thermal`：温控压力；
- `cpu_power`：CPU 采样。

如果当前 macOS 版本的 `powermetrics` 不接受某个 sampler，保留能正常工作的组合，并在 `manifest` 和报告中注明缺失项；不要用 `0` 替代缺失字段。

> `powermetrics` 的功耗是平台估算值，适合观察趋势和同机对比，不适合当作跨设备绝对性能结论。

## 3. 启动 `monitor_machine.py`

在另一个终端窗口启动监控。`--duration` 应覆盖整个评测过程，建议比预期 bench 时长多留 2～5 分钟：

```bash
python tools/monitor_machine.py \
  --duration 21600 \
  --interval 2 \
  --disk disk0 \
  --powermetrics-file /Users/jiuwen/powermetrics_qwen38.txt \
  --follow-powermetrics \
  --workload-counters-file /Users/jiuwen/LLMBenchmark/results/mac_benchmark/monitor/qwen38-live/workload_counters.jsonl \
  --workload-window-s 15 \
  --theoretical-tflops <理论峰值，可为空> \
  --official-bandwidth-gb-s <理论统一内存带宽，可为空> \
  --prefill-flops-per-token <每输入 token 的 FLOPs> \
  --decode-flops-per-token <每输出 token 的 FLOPs> \
  --prefill-bytes-per-token <每输入 token 的字节数> \
  --decode-bytes-per-token <每输出 token 的字节数> \
  --quantization Q4_K_M \
  --label qwen38-c1-c16 \
  --out /Users/jiuwen/LLMBenchmark/results/mac_benchmark/monitor/qwen38-live \
  --chart
```

如果暂时没有可靠的 FLOPs/token 或 bytes/token，可以不传这些参数；硬件实时状态仍会记录，相关有效算力/带宽字段保持 `null`。

首次新跑时不要加：

```text
--start-at-end
--start-workload-at-end
```

`--start-at-end` 用于附着已有的 `powermetrics` 文件，`--start-workload-at-end` 用于中途接入已有的 workload 文件。加在新鲜运行上会跳过早期样本。

## 4. 启动 bench 并写入实时 workload

推荐正式入口使用 `tools/bench_accuracy_concurrency.py`。bench 每完成一个成功请求，就向 `--live-counters-file` 追加一行 JSONL 并立即 flush，monitor 增量跟随该文件。

```bash
python tools/bench_accuracy_concurrency.py \
  --engine splash-qwen36 \
  --query-source manifest \
  --manifest /Users/jiuwen/LLMBenchmark/config/qwen36_sample_882.jsonl \
  --concurrency 1,2,4,8,16 \
  --service-max-concurrency 16 \
  --mode stream \
  --max-output-tokens 2048 \
  --live-counters-file /Users/jiuwen/LLMBenchmark/results/mac_benchmark/monitor/qwen38-live/workload_counters.jsonl \
  --workload-window-s 15 \
  --theoretical-tflops <理论峰值，可为空> \
  --official-bandwidth-gb-s <理论统一内存带宽，可为空> \
  --prefill-flops-per-token <每输入 token 的 FLOPs> \
  --decode-flops-per-token <每输出 token 的 FLOPs> \
  --prefill-bytes-per-token <每输入 token 的字节数> \
  --decode-bytes-per-token <每输出 token 的字节数> \
  --out /Users/jiuwen/LLMBenchmark/results/mac_benchmark/accuracy/qwen38-live
```

注意：

1. `--live-counters-file` 推荐使用**绝对路径**。相对路径会被 bench 解释为 `--out` 目录下的相对路径，monitor 侧容易拿错文件。
2. `monitor_machine.py` 与 `bench_accuracy_concurrency.py` 必须指向同一个绝对路径。
3. bench 启动时会删除同名旧 live counter 文件并重新创建；monitor 能处理文件重建和截断，但正式评测仍建议每次使用新的运行目录。
4. 失败请求不会进入 workload JSONL，因此不会把超时/错误误算成低 token 吞吐；错误率仍由 bench 的阶段汇总记录。
5. `--workload-window-s` 必须与 bench/monitor 的口径一致，推荐 `15` 秒；窗口越短越敏感，越长越平滑。

## 5. 启动顺序与 tmux 注意事项

推荐顺序：

```text
1. sudo powermetrics ... -o <powermetrics.txt>
2. python tools/monitor_machine.py ... --follow-powermetrics ...
3. python tools/bench_accuracy_concurrency.py ... --live-counters-file <同一个 workload JSONL>
```

原因：

- 先启动 `powermetrics`，monitor 才有 GPU/NPU/thermal 数据可跟随；
- 先启动 monitor，bench 创建 workload 文件后可以从第一个请求开始读取；
- 最后启动 bench，避免评测开始后才发现监控路径或权限错误。

tmux 示例：

```bash
# 窗口 1
tmux new -s qwen38-power 'sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal -i 2000 -o /Users/jiuwen/powermetrics_qwen38.txt'

# 窗口 2
tmux new -s qwen38-monitor 'cd /Users/jiuwen/LLMBenchmark && python tools/monitor_machine.py ...'

# 窗口 3
tmux new -s qwen38-bench 'cd /Users/jiuwen/LLMBenchmark && python tools/bench_accuracy_concurrency.py ...'
```

不要在 `sudo powermetrics` 的前台终端里同时运行长评测；使用 tmux 或独立终端，避免窗口关闭导致采集中断。若使用 `nohup` 启动，必须把 stdout/stderr 写入日志并确认进程仍在运行。

## 6. workload JSONL schema

每个成功且带 token 数据的请求对应一行：

```json
{
  "wall_time": 1790000000.123,
  "concurrency": 4,
  "dataset_id": "gpqa",
  "record_index": 12,
  "mode": "stream",
  "prompt_tokens": 512,
  "completion_tokens": 128,
  "ttft_s": 0.42,
  "prefill_s": 0.38,
  "decode_s": 4.12,
  "e2e_s": 4.54
}
```

字段语义：

| 字段 | 含义 |
|---|---|
| `wall_time` | 请求完成时刻，Unix epoch 秒；monitor 用它做时间对齐 |
| `concurrency` | 当前并发档位 |
| `dataset_id` / `record_index` | 题目定位，便于回溯到原始样本 |
| `mode` | `stream` 或 `nonstream`，不同模式不得混算 |
| `prompt_tokens` | 服务端返回的输入 token 数 |
| `completion_tokens` | 服务端返回的输出 token 数 |
| `ttft_s` | 请求开始到首 token 的时间 |
| `prefill_s` / `decode_s` | 可用时记录的分阶段服务端时间 |
| `e2e_s` | 请求总耗时 |

缺失字段使用 `null`，不要补 `0`。monitor 只消费完整 JSON 行；文件轮转、截断或 bench 重建同名文件时会重新定位。

## 7. 滑动窗口统计的限制

`monitor_machine.py` 默认按最近 15 秒内**已完成请求**统计 token 吞吐。必须理解以下限制：

- 长请求还没完成时不会进入窗口，图上可能暂时偏低；
- 并发时 Prefill 和 Decode 可能重叠，分阶段值不是严格的串行拆分；
- `input_tokens_per_s` 和 `output_tokens_per_s` 是服务端返回 token 的聚合吞吐；
- 窗口内没有有效事件时，实时推导值返回 `null`，不会沿用旧值伪造当前状态；
- 不同并发档位不要合并成一个平均值；每档应单独出图、单独汇总；
- `effective_bandwidth_gb_s` 是聚合估计，不是硬件总线带宽计数器；
- `effective_tflops` 依赖显式 FLOPs/token；没有可靠参数时宁可留空；
- MoE 的 bytes/token 必须按实际激活专家路径估算，不能直接用总参数乘量化字节数。

如果需要更细的阶段分析，优先使用请求记录中的 `prefill_s`、`decode_s`、`ttft_s` 和 `decode_tok_per_s`，再与 monitor 的有效值交叉核对。

## 8. MoE 模型的特殊口径

对 Qwen3.6-35B-A3B、Gemma-4-26B-A4B、Qwen3.8-Flash-Next 等 MoE/稀疏激活模型：

- 不能把 `35B` 总参数直接乘以每参数字节数当作每 token 激活字节数；
- `decode_flops_per_token` 和 `decode_bytes_per_token` 应按实际激活专家、路由 top-k、共享专家和 KV 读取路径估算；
- `prefill_flops_per_token` 还受输入长度、attention 实现和批处理策略影响；
- expert/权重是否从 SSD 流式读取，要用 `disk_read_mib_s` 和引擎日志交叉确认；
- 报告必须写明估算公式、参数来源、量化格式和是否包含 KV cache。

## 9. 输出文件与报告写法

一次监控通常产生：

```text
<out>/
  metrics.jsonl       # 每次采样的完整机器状态 + compute 推导值
  manifest.json       # 参数、理论基线、workload 路径、运行标签
  charts/             # --chart 时的 PNG
```

报告中建议分成三张表：

1. **硬件实时状态表**：CPU、统一内存、swap、GPU/NPU residency/frequency/power、thermal pressure、SSD；
2. **workload 有效值表**：input/output tok/s、Prefill/Decode effective TFLOPS、effective bandwidth、利用率；
3. **理论基线表**：理论峰值、来源、利用率分母和限制说明。

推荐统一表述：

> `powermetrics` 可实时采集 GPU/NPU 利用率、active frequency、GPU/ANE power、thermal pressure；不能直接采集实时物理 TFLOPS 或 GPU/NPU 物理总线带宽计数器。项目中的 `effective_tflops` / `effective_bandwidth_gb_s` 是由完成请求 token 吞吐和显式参数推导的 workload-level 值，不是硬件总线计数器。

## 10. 最小验收清单

- [ ] `powermetrics` 文件在评测期间持续增长；
- [ ] `monitor_machine.py` 的 `metrics.jsonl` 在评测期间逐行增长；
- [ ] workload JSONL 在每次成功请求后出现新行；
- [ ] monitor 和 bench 使用同一个绝对 `--live-counters-file`；
- [ ] `manifest.json` 记录理论基线、FLOPs/token、bytes/token 和量化格式；
- [ ] 缺失的 GPU/NPU/thermal/算力/带宽字段为 `null`，没有补 `0`；
- [ ] 每个并发档位单独出图和汇总；
- [ ] 报告中区分硬件实时状态、workload 有效值、理论基线；
- [ ] 失败请求保留在 bench 请求记录中，但不进入 token 吞吐；
- [ ] 若出现 swap、OOM、热降频或 SSD 流式读取，在结论中单列说明。
