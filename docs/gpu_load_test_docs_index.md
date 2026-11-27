# GPU 负载测试相关产物索引

## 1. 压测脚本与工具

- `tools/gpu_power_ceiling_probe.py`：GPU 压载脚本
- `tools/analyze_gpu_utilization.py`：GPU 利用率/full_load 统计分析
- `tools/plot_gpu_cpu_utilization.py`：CPU/GPU utilization + full_load 滚动百分比图
- `tools/plot_gpu_load_alternatives.py`：两种连续 GPU 负载指标对比图
- `tools/plot_gpu_cpu_power_utilization.py`：CPU + GPU residency + GPU power utilization 图
- `tools/gpu_metrics_stream.py`：powermetrics 流式解析，输出 monitor 兼容 JSONL

## 2. 压测原始数据

- `results/powermetrics_gpu_ceiling.txt`

## 3. 2048 结果图

- `docs/qwen38-27b-20261126-092930-stream-shard-2048_gpu_cpu_utilization.png`
- `docs/qwen38-27b-20261126-092930-stream-shard-2048_gpu_cpu_utilization_full_load.png`
- `docs/qwen38-27b-20261126-092930-stream-shard-2048_gpu_cpu_power_utilization.png`
- `docs/qwen38-27b-20261126-092930-stream-shard-2048_gpu_effective_load.png`
- `docs/qwen38-27b-20261126-092930-stream-shard-2048_gpu_power_utilization.png`

## 4. 报告

- `docs/qwen38-27b-20261126-092930-stream-shard-2048_GPU_CPU_full_load分析.md`
- `docs/qwen38-27b-20261126-130755-stream-4096-shard_GPU利用率报告_实测ceiling.md`

## 5. power ceiling 测试结论

```text
power_ceiling_w = 37.4 W
GPU active residency P50/P95 = 100% / 100%
GPU power P50/P95 = 31.6W / 36.2W
GPU power 10s rolling max = 37.4W
GPU power instantaneous max = 70.1W
```
