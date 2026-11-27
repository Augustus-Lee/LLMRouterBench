# GPU 有效算力统计

> flops_per_token = 5.400000e+10
> peak_tflops = null

| 并发 | wall_tflops | decode_mean_tflops | prefill_mean_tflops | compute_utilization_pct(value/tflops) |
|---|---:|---:|---:|---:|
| 1 | 0.258 | 3.656 | - | - |
| 2 | 0.586 | 3.394 | - | - |
| 4 | 0.607 | 9.067 | - | - |
| 8 | 0.799 | 12.684 | - | - |

## 总体按 wall time 汇总

- total_wall_s = 8516.68
- total_tokens = 75386
- tokens_per_s = 8.852
- effective_tflops = 0.4780
- compute_utilization_pct = -
