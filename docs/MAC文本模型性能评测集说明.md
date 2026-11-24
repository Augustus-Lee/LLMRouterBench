# Mac 文本模型性能评测集说明（已弃用）

> ⚠️ 2026-11-24 起，本轮评测改用仓库原生能力评测集，不再使用本文件描述的性能 workload。

## 当前正式评测集

正式 query 使用仓库原生能力评测集，配置入口为 `config/edge_router_config.yaml` 的 `datasets` 段：

| dataset_id | 数据文件 | 全量条数 | 判分方式 |
|---|---|---:|---|
| `gpqa` | `data/GPQA/gpqa_diamond.json` | 198 | 选项匹配 |
| `mmlupro` | `data/MMLUPro/test.json` | 1001 | 选项匹配 |
| `math500` | `data/MATH500/test.json` | 500 | 答案抽取 + 数学判分 |
| `arcc` | `data/arc_c/arc_test.json` | 1172 | 选项匹配 |
| `gsm8k` | `data/GSM8K/test.json` | 100 | 数值匹配 |
| `mmlu` | `data/MMLU/test.json` | 182 | 选项匹配 |

正式方案见：

- `docs/MAC_文本模型性能评测方案.md`
- `docs/MAC文本模型性能评测方案V2.md`

`benchmark_data/performance/` 下生成的三层性能 workload 仅作为历史/生成器产物保留，不作为本轮正式评测集。
