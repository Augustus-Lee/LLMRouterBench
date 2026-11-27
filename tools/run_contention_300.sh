#!/bin/bash
# 资源竞争正式评测：CPU / GPU / 带宽 各 4 档（20/40/60/80%），每档 24 题（≈25）。
#
# 方案口径（300 题）：
#   - 维度：cpu / gpu / bandwidth 三组，每组 4 档 = 100 题/维，共 ~288 题（6 数据集 × 4 = 24 题/档）。
#   - CPU 档 = 整机 CPU used%（闭环 PID）；GPU 档 = 访存绑定 2048×2048 kernel 的 duty%；
#     带宽档 = 目标理论带宽 273 GB/s 的百分比。
#   - 固定并发 1；0% 基线复用旧矩阵（results/contention_qwen38_20261127_105102）。
#   - 内存 6 GiB 固定占用不在此脚本（独立 baseline，如需要另设 stressor）。
#   - 同一 seed=42 + limit-per-dataset=4 → 12 个档位复用同一批 24 题，跨档可比。
#   - 档位切换 = 题数驱动：每档跑满 24 题后自动切下一档（非定时）。
#
# 依赖：
#   - splash 服务在 127.0.0.1:18080 在线（模型 Qwen/Qwen3.8-27B）。
#   - 用户手动启动的 powermetrics 正在写 /Users/jiuwen/powermetrics_qwen38.txt。
#   - 带宽校准曲线：results/bandwidth_calibration_clean.json（已生成）。
#
# 时间估算（max-output-tokens 10000，实际输出长度按题而定）：
#   每档 24 题、共 12 档；总时长取决于答案长度，粗略 2-4 小时。
set -u
cd "$HOME/AA"

TS=$(date +%Y%m%d_%H%M%S)
OUT="$HOME/AA/results/contention_300_$TS"
mkdir -p "$OUT"

PY="$HOME/AA/.venv-314/bin/python"

nohup "$PY" tools/run_contention_bench.py \
  --out "$OUT" \
  --python "$PY" \
  --resources cpu,gpu,bandwidth \
  --tiers 20,40,60,80 \
  --bench-args "--engine splash-qwen38 --query-source data --concurrency 1 --mode stream --limit-per-dataset 4 --max-output-tokens 10000 --temperature 0.0 --seed 42" \
  --monitor-args "--powermetrics-file /Users/jiuwen/powermetrics_qwen38.txt --follow-powermetrics --start-at-end --gpu-max-freq-mhz 1620" \
  --theoretical-bandwidth-gb-s 273 \
  --calibration-file "$HOME/AA/results/bandwidth_calibration_clean.json" \
  > "$OUT/run.log" 2>&1 &

PID=$!
echo "$PID" > "$OUT/run.pid"
echo "$OUT" > "$HOME/AA/results/.last_contention_300"
echo "OUT=$OUT PID=$PID"