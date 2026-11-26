# Mac 文本模型正式评测 SOP

> 适用机器：`192.168.3.5` / `macmini-m5-jiuwen` / Apple M5 Pro / 64GB 统一内存  
> 远端仓库：`/Users/jiuwen/AA`  
> 适用脚本：`tools/bench_accuracy_concurrency.py`、`tools/monitor_machine.py`、`tools/sample_882.py`  
> 评测口径：固定 882 题 manifest；并发阶梯 `1 → 2 → 4 → 8`；跨档切片合计覆盖完整 882 题  
> 文档版本：2026-11-26  
> 本轮修订：Python 3.14 专用环境、四档跨档切片、monitor 字段修复记录  
> 关联方案：[MAC_文本模型性能评测方案.md](./MAC_文本模型性能评测方案.md)、[MAC文本模型性能评测方案V2.md](./MAC文本模型性能评测方案V2.md)、[MAC_算力带宽实时监控接入.md](./MAC_算力带宽实时监控接入.md)  
> 结论口径：报告只能写“固定 882 题抽样评测”，不能写“全量评测”。

---

## 1. 目的、范围与不可突破的边界

### 1.1 目的

本 SOP 规定 Mac mini 上本地文本大模型**正式性能评测**的可复现流程，覆盖：

- 评测窗口、机器基线、Python 环境与模型资产确认；
- 推理服务启动、冷加载计时、接口探活；
- `powermetrics` 与 `monitor_machine.py` 三层指标监控；
- 单请求探活、六题冒烟、manifest dry-run；
- 非流式与流式正式评测；
- 并发阶梯、熔断、人工巡检、异常止损；
- 产物验收、收尾、分引擎报告口径。

### 1.2 第一轮与第二轮对象

第一轮只测以下三个模型：

| 表 | 模型 | 引擎 | 权重/量化 | `--engine` | 端口 |
|---|---|---|---|---|---:|
| A | Qwen3.8-27B | Splash 1.5.0 + DFlash2 | GGUF Q4_K_M | `splash-qwen38` | 18080 |
| B | Qwen3.6-35B-A3B | Splash 1.5.0 + DFlash2 | MLX 4bit affine | `splash-qwen36` | 18081 |
| C | Gemma-4-26B-A4B | MLX-VLM 0.8.0 | MLX 4bit affine | `mlxvlm-gemma` | 18091 |

第二轮实验对象 `Qwen3.8-Flash-Next` 使用端口 18082，单独成表，不参与第一轮排名。其前置条件、启动边界见第 13.4 节。

### 1.3 指标分层

正式评测必须同时保留三层指标：

| 层级 | 内容 | 采集来源 |
|---|---|---|
| L1 硬件实时状态 | CPU、统一内存、进程 RSS、wired、swap、GPU/ANE residency/frequency/power、thermal pressure、SSD、冷加载 | `monitor_machine.py`、`powermetrics`、bench 的 `SystemMonitor`、启动脚本 |
| L2 workload 指标 | TTFT、Prefill tok/s、Decode tok/s、请求吞吐、输入/输出 token 吞吐、E2E、成功率、错误率、准确率、`finish_reason` | `bench_accuracy_concurrency.py` |
| L3 理论基线与推导值 | 理论峰值算力、官方统一内存带宽、每 token FLOPs/bytes、workload-level 有效算力/有效带宽与利用率 | bench/monitor 显式参数计算 |

> `effective_tflops` 和 `effective_bandwidth_gb_s` 是由已完成请求的 token 吞吐与显式参数推导的 workload-level 有效值，**不是 Apple GPU/ANE 物理计数器**。MoE 必须按实际激活专家路径估算，不能直接用总参数量乘量化字节数。缺少可靠来源时写 `null`，不能补 `0`。

### 1.4 不可突破的边界

- 不把不同引擎、量化、上下文或 KV/state 精度合并成一个总排名；
- 不把并发吞吐与单请求延迟混成一个指标；
- 不把冷加载与热态请求混算；
- 不删除失败样本，不用 `0` 或猜测值补齐 GPU/NPU/thermal/算力/带宽字段；
- 不为追求耗时而默认使用跨档切片；正式主矩阵默认每档跑完整 882 题；
- 不在共享推理负载未释放或隔离时宣称正式 tok/s。

---

## 2. 角色、终端与固定执行顺序

### 2.1 角色

| 角色 | 职责 |
|---|---|
| 评测执行人 | 启动服务、测量冷加载、运行 bench、记录阶段结果 |
| 监控巡检人 | 检查 powermetrics/monitor、RSS/swap/温控、服务日志与熔断信号 |

单人执行时也必须使用多个终端窗口，避免前台命令互相阻塞。

### 2.2 登录与目录

```bash
ssh macmini-m5-jiuwen
cd /Users/jiuwen/AA
```

### 2.3 tmux 四窗口

```bash
tmux new -s macbench
```

| 窗口 | 用途 |
|---|---|
| `0:service` | 启动/停止推理服务，查看服务日志 |
| `1:power` | `sudo powermetrics` 持续采样 |
| `2:monitor` | 按阶段运行 `monitor_machine.py` |
| `3:bench` | 探活、冒烟、dry-run、非流式/流式正式评测 |

切换窗口：`Ctrl-b 0`、`Ctrl-b 1`、`Ctrl-b 2`、`Ctrl-b 3`。

### 2.4 固定执行顺序

```text
机器基线与独占检查
→ 创建 RUN_ROOT / run.env
→ 校验 sample_882.jsonl
→ 启动 powermetrics（全轮一次）
→ 启动 service-smoke 阶段 monitor
→ 记录 T0，启动推理服务
→ 等待 ready，计算 cold_load_s
→ 真实 Chat Completions 探活
→ 六题冒烟，停止 service-smoke monitor
→ manifest dry-run
→ 启动 nonstream monitor，跑非流式矩阵，停止 monitor
→ 启动 stream monitor，跑流式矩阵，停止 monitor
→ 停止服务，停止 powermetrics，验收产物
```

> 非流式与流式的先后顺序可按实际需要交换（本轮实际先跑了流式），但两轮必须使用不同的 monitor 目录、workload 文件和输出目录，不能复用。

每个 tmux 窗口在操作前都先执行：

```bash
source "$RUN_ROOT/run.env"
```

---

## 3. 前置检查

### 3.1 Python 环境：必须 >= 3.10

这是硬要求。`evaluation/factory.py` 使用 `str | Benchmark` 类型语法，Python 3.9 无法解析。

当前远端已有专用 Python 3.14 环境，**不要重复 `python -m venv`**：

```bash
export BENCH_PYTHON='/Users/jiuwen/AA/.venv-314/bin/python'
```

该环境已经安装并验证过以下正式评测所需依赖：

```text
openai / psutil / datasets / matplotlib
loguru / httpx / tenacity / tiktoken / pylatexenc
pyyaml / tqdm / numpy / pandas / sympy / nltk
```

正式运行前只做验证，不做重建或升级无关依赖：

```bash
"$BENCH_PYTHON" --version
"$BENCH_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
"$BENCH_PYTHON" -c 'import openai, psutil, datasets, matplotlib, loguru, httpx, tenacity, tiktoken, pylatexenc'
"$BENCH_PYTHON" -c 'from evaluation.factory import EvaluatorFactory; EvaluatorFactory()'
"$BENCH_PYTHON" -m compileall -q evaluation tools baselines
```

正式运行前必须确认：

- Python 为 3.14.8；
- 上表中的依赖可导入；
- `EvaluatorFactory` 可实例化；
- `compileall` 退出码为 0。

说明：

- 远端旧 `.venv-bench` 是 Python 3.9.6，**不得用于正式评测**；
- 不要在正式命令中依赖不确定的裸 `python3`；
- 如果某个依赖缺失，只安装到 `.venv-314`，不要污染系统 Python；
- 每轮开始仍要运行上述验证并保存输出，不能只凭“上次能用”下结论。

把版本与依赖信息写入本轮 metadata：

```bash
"$BENCH_PYTHON" - <<'PY_VER'
import sys
import openai
import psutil
import datasets
import matplotlib
print("python", sys.version.split()[0])
print("openai", openai.__version__)
print("psutil", psutil.__version__)
print("datasets", datasets.__version__)
print("matplotlib", matplotlib.__version__)
PY_VER
```

### 3.2 代码、分支与脚本

```bash
cd /Users/jiuwen/AA
git status --short --branch
git log -1 --oneline
git fetch origin codex/live-hardware-metrics
git restore --source origin/codex/live-hardware-metrics -- tools/monitor_machine.py tools/bench_accuracy_concurrency.py
ls -lh tools/bench_accuracy_concurrency.py \
       tools/monitor_machine.py \
       tools/sample_882.py
```

正式运行前确认：

- 两个核心脚本已经同步到 `origin/codex/live-hardware-metrics`；
- `results/sample_882.jsonl` 已同步到远端；
- `monitor_machine.py` 必须保留本节末尾列出的 GPU/ANE/swap/thermal 热修，不要用旧 main 版本覆盖；
- 工作区未提交文件不影响本次运行；
- 固定本次运行的 commit 或 digest。

> 本 SOP 只规定本地文档，不自动提交或推送代码。远端若存在未提交改动，先核对来源，不能擅自丢弃。

### 3.3 机器基线

```bash
sw_vers
sysctl -n hw.model
sysctl -n hw.memsize
sysctl -n hw.ncpu
df -h /System/Volumes/Data
memory_pressure | tail -20
vm_stat | head -20
sysctl vm.swapusage
```

检查共享推理负载与端口：

```bash
ps -axo pid,rss,command | egrep 'ollama|mlx_lm|mlx_vlm|splash serve|llama-server' | egrep -v egrep
lsof -nP -iTCP -sTCP:LISTEN | egrep ':(8080|8090|11434|11435|18080|18081|18082|18091)\b' || true
```

正式测速必须满足：

- 单模型、单实例；
- Ollama、MLX-LM、其他 `llama-server` 推理负载已停止或隔离；
- 测试窗口内禁止下载、复制大文件、索引构建和备份；
- `18080/18081/18082/18091` 无冲突；
- 未取得机器使用方同意，不得直接停止共享服务。

机器基线写入 `RUN_ROOT/baseline/`：

```bash
mkdir -p "$RUN_ROOT/baseline"
{
  ps -axo pid,rss,command
  lsof -nP -iTCP -sTCP:LISTEN
  memory_pressure
  vm_stat
  sysctl vm.swapusage
  df -h /System/Volumes/Data
} > "$RUN_ROOT/baseline/preflight.txt" 2>&1
```

---

## 4. 一次正式运行的目录、环境与样本

### 4.1 创建 `RUN_ROOT` 与 `run.env`

每次运行必须使用唯一 `RUN_ID`。先定义解释器绝对路径、模型标签、引擎、端口、服务日志和三类 workload 路径，再生成 `run.env`。

以下示例以 Qwen3.8-27B 为例。执行前把 `BENCH_PYTHON` 改成现场验证过的绝对路径：

```bash
export BENCH_PYTHON='/Users/jiuwen/AA/.venv-314/bin/python'
export MODEL_TAG='qwen38-27b'
export ENGINE='splash-qwen38'
export API_MODEL='Qwen/Qwen3.8-27B'
export PORT='18080'
export BASE_URL='http://127.0.0.1:18080/v1'
export QUANTIZATION='Q4_K_M'

export RUN_ID="${MODEL_TAG}-$(date +%Y%m%d-%H%M%S)"
export RUN_ROOT="/Users/jiuwen/AA/results/mac_benchmark/runs/$RUN_ID"
export POWER_FILE="$RUN_ROOT/monitor/powermetrics.txt"
export SERVICE_MONITOR_DIR="$RUN_ROOT/monitor/service-smoke"
export NONSTREAM_MONITOR_DIR="$RUN_ROOT/monitor/nonstream"
export STREAM_MONITOR_DIR="$RUN_ROOT/monitor/stream"
export SMOKE_WORKLOAD_FILE="$SERVICE_MONITOR_DIR/workload_counters.jsonl"
export NONSTREAM_WORKLOAD_FILE="$NONSTREAM_MONITOR_DIR/workload_counters.jsonl"
export STREAM_WORKLOAD_FILE="$STREAM_MONITOR_DIR/workload_counters.jsonl"
export SERVER_PID_FILE="$RUN_ROOT/logs/splash-qwen38.pid"
export SERVER_LOG="$RUN_ROOT/logs/splash-qwen38.log"

mkdir -p "$RUN_ROOT"/{logs,baseline,smoke,accuracy/nonstream,accuracy/stream,monitor/service-smoke,monitor/nonstream,monitor/stream}

cat > "$RUN_ROOT/run.env" <<EOF
# 由本 SOP 生成；不要写成 export PY=\${PY:?...} 的递归形式。
export BENCH_PYTHON='$BENCH_PYTHON'
export PY="$BENCH_PYTHON"
export RUN_ID='$RUN_ID'
export RUN_ROOT='$RUN_ROOT'
export POWER_FILE='$POWER_FILE'
export SERVICE_MONITOR_DIR='$SERVICE_MONITOR_DIR'
export NONSTREAM_MONITOR_DIR='$NONSTREAM_MONITOR_DIR'
export STREAM_MONITOR_DIR='$STREAM_MONITOR_DIR'
export SMOKE_WORKLOAD_FILE='$SMOKE_WORKLOAD_FILE'
export NONSTREAM_WORKLOAD_FILE='$NONSTREAM_WORKLOAD_FILE'
export STREAM_WORKLOAD_FILE='$STREAM_WORKLOAD_FILE'
export MODEL_TAG='$MODEL_TAG'
export ENGINE='$ENGINE'
export API_MODEL='$API_MODEL'
export PORT='$PORT'
export BASE_URL='$BASE_URL'
export QUANTIZATION='$QUANTIZATION'
export SERVER_PID_FILE='$SERVER_PID_FILE'
export SERVER_LOG='$SERVER_LOG'
EOF

chmod 600 "$RUN_ROOT/run.env"
source "$RUN_ROOT/run.env"
```

> `run.env` 里的 `PY` 必须展开为已核验的绝对路径，不能自引用尚未设置的变量。每个新 tmux 窗口都先 `source "$RUN_ROOT/run.env"`。

**禁止复用旧 `RUN_ROOT`**：bench 启动时会删除同名 `--live-counters-file`。冒烟、非流式、流式使用不同目录和不同 workload 文件，不能共用同一个 `workload_counters.jsonl`。

### 4.2 复制并校验固定 882 题 manifest

```bash
cp /Users/jiuwen/AA/results/sample_882.jsonl "$RUN_ROOT/sample_882.jsonl"
shasum -a 256 "$RUN_ROOT/sample_882.jsonl" | tee "$RUN_ROOT/sample_882.sha256"
wc -l "$RUN_ROOT/sample_882.jsonl"
```

本次固定清单的 SHA256 应为：

```text
52208bf7eac6f4c53cb99daf69b200db272e3220978ebdca501cbe2a771f2456
```

分集数量核对：

```bash
awk -F'"' '/dataset_id/{c[$4]++} END {for(k in c) print k,c[k]}' \
  "$RUN_ROOT/sample_882.jsonl" | sort
```

期望：

```text
arcc 150
gpqa 150
gsm8k 100
math500 150
mmlu 182
mmlupro 150
```

如清单需要重建，必须固定 seed 并记录新哈希：

```bash
"$PY" tools/sample_882.py \
  --seed 42 \
  --out "$RUN_ROOT/sample_882.jsonl"
```

### 4.3 样本口径

| dataset_id | 固定样本数 |
|---|---:|
| `gpqa` | 150 |
| `mmlupro` | 150 |
| `math500` | 150 |
| `arcc` | 150 |
| `gsm8k` | 100 |
| `mmlu` | 182 |
| **合计** | **882** |

所有模型共用同一份清单，按 `(dataset_id, record_index)` 严格对齐。

> `gsm8k` 和 `mmlu` 的抽样数等于其当前数据总量；其余数据集是固定抽样。正式报告必须写“固定 882 题抽样评测”，不能写成“全量评测”。

---

## 5. 服务启动、冷加载与探活

### 5.1 启动前检查

```bash
source "$RUN_ROOT/run.env"
lsof -nP -iTCP:"$PORT" -sTCP:LISTEN || true
ps -axo pid,rss,command | rg 'splash serve|mlx_vlm.server' || true
```

确认目标端口空闲、无同模型残留进程后，再记录 T0 并启动服务。

### 5.2 冷加载定义

冷加载严格定义为：

```text
服务进程启动 T0 → `/ready` 或 `/health(model_ready=true)` 首次成功
```

要求：

- T0 必须在真正执行 `splash serve` / `mlx_vlm.server` 之前记录；
- Splash 使用 `/ready`；Gemma 使用 `/health` 并确认 `model_ready=true`；
- ready 轮询分辨率约 1 秒，报告中必须注明；
- ready 超时、进程提前退出或日志出现 OOM/Engine failed 时，终止本轮评测。

启动监听函数：

```bash
record_service_t0() {
  "$PY" - <<'PY_T0'
import time
print(f"{time.time():.6f}")
PY_T0
}

wait_splash_ready() {
  local port="$1"
  local deadline_s="${2:-600}"
  local i
  for i in $(seq 1 "$deadline_s"); do
    if ! kill -0 "$(cat "$SERVER_PID_FILE")" 2>/dev/null; then
      echo "服务进程已退出，ready 失败" >&2
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/ready" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_gemma_ready() {
  local port="$1"
  local deadline_s="${2:-600}"
  local i health
  for i in $(seq 1 "$deadline_s"); do
    if ! kill -0 "$(cat "$SERVER_PID_FILE")" 2>/dev/null; then
      echo "服务进程已退出，ready 失败" >&2
      return 1
    fi
    health="$(curl -fsS "http://127.0.0.1:${port}/health" 2>/dev/null || true)"
    if printf '%s' "$health" | "$PY" -c \
      'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("model_ready") is True else 1)' \
      2>/dev/null; then
      printf '%s' "$health" | tee "$RUN_ROOT/logs/health_ready.json"
      return 0
    fi
    sleep 1
  done
  return 1
}

finalize_cold_load() {
  local t0="$1"
  local t1
  t1="$("$PY" - <<'PY_T1'
import time
print(f"{time.time():.6f}")
PY_T1
)"
  echo "$t1" | tee "$RUN_ROOT/logs/ready_t1.txt"
  "$PY" - "$t0" "$t1" <<'PY_COLD' | tee "$RUN_ROOT/logs/cold_load_s.txt"
import sys
print(f"{float(sys.argv[2]) - float(sys.argv[1]):.3f}")
PY_COLD
}
```

### 5.3 Qwen3.8-27B 启动

在 `0:service` 窗口：

```bash
source "$RUN_ROOT/run.env"
T0="$(record_service_t0)"
echo "$T0" | tee "$RUN_ROOT/logs/service_t0.txt"

nohup /opt/homebrew/bin/splash serve \
  --model 'Qwen/Qwen3.8-27B' \
  --model-dir /Users/jiuwen/Models/splash/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --draft-model /Users/jiuwen/Models/splash/Qwen3.8-27B-DFlash2 \
  --language-only \
  --offline \
  --host 127.0.0.1 \
  --port 18080 \
  --max-context 32K \
  --max-memory 40G \
  --kv-format int8 \
  --state-format bf16 \
  --verify-rows all \
  --queue-size 8 \
  --request-timeout 30m \
  --idle-release 10m \
  > "$SERVER_LOG" 2>&1 < /dev/null &

echo $! | tee "$SERVER_PID_FILE"

if ! wait_splash_ready 18080 600; then
  echo "Splash 在 600 秒内未 ready，停止后续步骤" >&2
  tail -100 "$SERVER_LOG" >&2 || true
  exit 1
fi
finalize_cold_load "$T0"
curl -fsS http://127.0.0.1:18080/ready | tee "$RUN_ROOT/logs/ready.json"
```

注意：

- `--queue-size 8` 必须与当前四档阶梯最大并发 `8` 一致；如果后续恢复五档 `16`，再同步改回 `16` 并写入报告；
- Qwen3.8 与 Qwen3.6 的 DFlash2 不可互换；
- `--max-context` 和 `--max-memory` 是本次固定参数，必须原样写入报告；本条示例沿用方案 V2 的 32K/40G 口径。

### 5.4 Qwen3.6-35B-A3B 启动

```bash
source "$RUN_ROOT/run.env"
T0="$(record_service_t0)"
echo "$T0" | tee "$RUN_ROOT/logs/service_t0.txt"

nohup /opt/homebrew/bin/splash serve \
  --model 'mlx-community/Qwen3.6-35B-A3B-4bit' \
  --model-dir /Users/jiuwen/Models/Qwen3.6-35B-A3B-4bit \
  --draft-model /Users/jiuwen/Models/splash/Qwen3.6-35B-A3B-DFlash2 \
  --language-only \
  --offline \
  --host 127.0.0.1 \
  --port 18081 \
  --max-context 32K \
  --max-memory 42G \
  --kv-format int8 \
  --state-format bf16 \
  --verify-rows all \
  --queue-size 8 \
  --request-timeout 30m \
  --idle-release 10m \
  > "$SERVER_LOG" 2>&1 < /dev/null &

echo $! | tee "$SERVER_PID_FILE"

if ! wait_splash_ready 18081 600; then
  echo "Splash 在 600 秒内未 ready，停止后续步骤" >&2
  tail -100 "$SERVER_LOG" >&2 || true
  exit 1
fi
finalize_cold_load "$T0"
curl -fsS http://127.0.0.1:18081/ready | tee "$RUN_ROOT/logs/ready.json"
```

### 5.5 Gemma-4-26B-A4B 启动

启动前在本轮 `run.env` 中设置：

```bash
export ENGINE='mlxvlm-gemma'
export API_MODEL='/Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit'
export PORT='18091'
export BASE_URL='http://127.0.0.1:18091/v1'
export QUANTIZATION='MLX-4bit'
export SERVER_PID_FILE="$RUN_ROOT/logs/gemma4-server.pid"
export SERVER_LOG="$RUN_ROOT/logs/gemma4-server.log"
```

如这些值与旧 `run.env` 不一致，重新生成本轮 `run.env`，不要跨模型复用。

```bash
source "$RUN_ROOT/run.env"
T0="$(record_service_t0)"
echo "$T0" | tee "$RUN_ROOT/logs/service_t0.txt"

nohup /Users/jiuwen/mlx-bench-venv/bin/mlx_vlm.server \
  --host 127.0.0.1 \
  --port 18091 \
  --model /Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit \
  --max-tokens 2048 \
  --max-num-seqs 8 \
  --log-level INFO \
  > "$SERVER_LOG" 2>&1 < /dev/null &

echo $! | tee "$SERVER_PID_FILE"

if ! wait_gemma_ready 18091 600; then
  echo "Gemma 在 600 秒内未 ready，停止后续步骤" >&2
  tail -100 "$SERVER_LOG" >&2 || true
  exit 1
fi
finalize_cold_load "$T0"
```

注意：MLX-VLM 的 `--max-tokens 2048` 是单次生成上限，不是上下文窗口；`--max-num-seqs 8` 必须与当前四档最大并发一致并写入报告。

### 5.6 探活

Splash：

```bash
source "$RUN_ROOT/run.env"
curl -fsS "http://127.0.0.1:${PORT}/ready" | tee "$RUN_ROOT/logs/ready.json"
curl -fsS "http://127.0.0.1:${PORT}/status" | tee "$RUN_ROOT/logs/status.json"
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | tee "$RUN_ROOT/logs/models.json"
```

Gemma：

```bash
source "$RUN_ROOT/run.env"
curl -fsS "http://127.0.0.1:${PORT}/health" | tee "$RUN_ROOT/logs/health.json"
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | tee "$RUN_ROOT/logs/models.json"
```

必须确认模型 ID、上下文、量化、KV/state 格式与启动参数一致。只检查端口监听不算探活完成。

真实 Chat Completions 探活：

```bash
source "$RUN_ROOT/run.env"
curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${API_MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"只回答：探活成功\"}],
    \"max_tokens\": 32,
    \"temperature\": 0,
    \"stream\": false
  }" \
  | tee "$RUN_ROOT/logs/single_request.json" \
  | "$PY" -m json.tool
```

必须同时保存响应和 HTTP 状态。若返回非 2xx、空回答、错误 `finish_reason` 或服务日志出现异常，停止正式评测。

---
## 6. 实时监控：`powermetrics` 与 `monitor_machine.py`

### 6.1 监控职责

| 组件 | 记录内容 | 是否实时 |
|---|---|---|
| `sudo powermetrics` | GPU/ANE residency、频率、功耗、thermal pressure | 是，约 2 秒采样 |
| `monitor_machine.py` | CPU、统一内存、swap、SSD、GPU/ANE 解析、workload 滑动窗口、有效算力/带宽、PNG 图表 | 是，按 `--interval` 采样 |
| bench 的 `SystemMonitor` | 推理进程峰值 RSS、wired、swap 增量、平均 SSD 读速 | 按 stage 汇总 |
| bench 请求明细 | TTFT、Prefill/Decode、token、E2E、判分、错误 | 每请求落盘 |

### 6.2 全轮只启动一次 `powermetrics`

在 `1:power` 窗口执行：

```bash
source "$RUN_ROOT/run.env"
sudo powermetrics \
  --samplers cpu_power,gpu_power,ane_power,thermal \
  -i 2000 \
  -b 1 \
  -o "$POWER_FILE"
```

说明：

- `-i 2000` 是约每 2 秒采样；
- `-b 1` 是 `--buffer-size 1`，表示按行刷新，便于 `--follow-powermetrics` 增量读取；
- `--samplers` **必须包含 `ane_power`**，否则 `npu_usage_percent` 会全为 `null`；
- 文件必须使用本轮绝对路径，不要用 `./` 相对路径，否则换目录后 `monitor_machine.py` 找不到文件；
- 如果本轮开始前 powermetrics 已经在运行并已有一个旧文件，可以继续使用该文件，但所有 monitor 阶段都要加 `--start-at-end`；
- 该进程应覆盖服务启动、冷加载、探活、冒烟、非流式、流式和收尾；
- 若当前 macOS 不接受某个 sampler，保留可工作的组合，并在 manifest 和报告中注明缺失项，不能补 0；
- `powermetrics` 的功耗是平台估算值，适合同机趋势比较，不适合作跨设备绝对性能结论。

### 6.3 三阶段 monitor

`monitor_machine.py` 一次只跟随一个 workload JSONL；bench 启动时会删除同名 live counters 文件。因而不允许三阶段共用一个 workload。必须使用：

```text
service-smoke → 六题冒烟
nonstream     → 非流式正式矩阵
stream        → 流式正式矩阵
```

在 `2:monitor` 窗口定义通用函数：

```bash
cd /Users/jiuwen/AA
source "$RUN_ROOT/run.env"

monitor_phase() {
  local out="$1"
  local workload="$2"
  local label="$3"
  shift 3

  "$PY" tools/monitor_machine.py \
    --duration 21600 \
    --interval 2 \
    --disk disk0 \
    --powermetrics-file "$POWER_FILE" \
    --follow-powermetrics \
    --workload-counters-file "$workload" \
    --workload-window-s 15 \
    --quantization "$QUANTIZATION" \
    --label "$label" \
    --out "$out" \
    "$@" \
    --chart
}
```

`--duration 21600` 是 6 小时上限；实际可在阶段结束或到达上限时停止。若预计超过 6 小时，按实际时长调大。

#### 阶段 1：service-smoke

在 **服务启动前** 启动：

```bash
monitor_phase \
  "$SERVICE_MONITOR_DIR" \
  "$SMOKE_WORKLOAD_FILE" \
  "$RUN_ID-service-smoke" \
  --start-at-end \
  2>&1 | tee "$RUN_ROOT/logs/monitor-service-smoke.log"
```

该阶段覆盖服务冷加载、探活和六题冒烟。冒烟结束后按 `Ctrl-C`，等待：

- `metrics.jsonl` 最终补写；
- `charts/` 内 PNG 生成；
- monitor 打印 `[monitor] done` 或正常中断提示。

#### 阶段 2：nonstream

先启动 monitor：

```bash
monitor_phase \
  "$NONSTREAM_MONITOR_DIR" \
  "$NONSTREAM_WORKLOAD_FILE" \
  "$RUN_ID-nonstream" \
  --start-at-end \
  2>&1 | tee "$RUN_ROOT/logs/monitor-nonstream.log"
```

然后到 `3:bench` 窗口启动非流式正式评测。bench 完成后回到 monitor 窗口按 `Ctrl-C`，等待收尾和图表生成。

#### 阶段 3：stream

```bash
monitor_phase \
  "$STREAM_MONITOR_DIR" \
  "$STREAM_WORKLOAD_FILE" \
  "$RUN_ID-stream" \
  --start-at-end \
  2>&1 | tee "$RUN_ROOT/logs/monitor-stream.log"
```

然后启动流式正式评测。bench 完成后停止 monitor。

### 6.4 监控参数的关键规则

- 阶段 1/2/3 都使用 `--start-at-end`：只要 powermetrics 文件是先于 monitor 启动的持续文件，就只读取 monitor 启动后的新 GPU/NPU 样本；
- **不要**给阶段 2/3 加 `--start-workload-at-end`，因为阶段 2/3 的 workload 文件是新文件，必须从头读取；
- monitor 与 bench 的 `--workload-counters-file` 必须是同一个**绝对路径**；
- 只有在 powermetrics 和 monitor 同时、且文件完全从本轮 T0 前重新创建时，阶段 1 才考虑省略 `--start-at-end`；
- 三个阶段不得共用 `metrics.jsonl` 或 `workload_counters.jsonl`；
- 每个阶段必须先启动 monitor，再启动 bench；
- `--quantization` 按模型设置：Qwen3.8 为 `Q4_K_M`；Qwen3.6/Gemma 为 `MLX-4bit`；
- `--theoretical-tflops`、`--official-bandwidth-gb-s`、每 token FLOPs/bytes 只在有可靠依据时传入；缺失时字段必须为 `null`；
- `--workload-window-s` 在 bench 和 monitor 两侧都使用 `15`。

### 6.5 可选：接入理论基线与有效算力/带宽

若已有经过确认的 M5 Pro 理论峰值、官方统一内存带宽以及本次模型的分阶段 FLOPs/bytes 参数，在 monitor 与 bench 命令中同时加入同一组参数：

```bash
--theoretical-tflops "$THEORETICAL_TFLOPS" \
--official-bandwidth-gb-s "$OFFICIAL_BANDWIDTH_GB_S" \
--prefill-flops-per-token "$PREFILL_FLOPS_PER_TOKEN" \
--decode-flops-per-token "$DECODE_FLOPS_PER_TOKEN" \
--prefill-bytes-per-token "$PREFILL_BYTES_PER_TOKEN" \
--decode-bytes-per-token "$DECODE_BYTES_PER_TOKEN"
```

报告必须同时写清：

```text
effective_tflops / effective_bandwidth_gb_s
= workload-level 由 token 吞吐推导
≠ Apple GPU/ANE 物理计数器
```

MoE（如 Qwen3.6-35B-A3B）必须按激活参数/激活专家路径估算；不能直接把 35B 总参数量乘量化字节数。没有可靠推导时宁可不填。

### 6.6 监控健康检查

每次阶段 monitor 启动后确认：

```bash
ls -lh "$POWER_FILE"
tail -3 "$POWER_FILE"
wc -l "$SERVICE_MONITOR_DIR/metrics.jsonl" 2>/dev/null || true
ls -lh "$SERVICE_MONITOR_DIR/manifest.json" 2>/dev/null || true
```

在服务启动和请求开始后确认文件持续增长：

```bash
wc -l "$SERVICE_MONITOR_DIR/metrics.jsonl"
wc -l "$SMOKE_WORKLOAD_FILE" 2>/dev/null || true
```

正式阶段同理检查 `$NONSTREAM_WORKLOAD_FILE`、`$STREAM_WORKLOAD_FILE`。文件不增长时不得继续正式评测。

---

### 6.7 已知坑与已修复字段

`monitor_machine.py` 在本轮实际运行中确认了以下问题，脚本已经修复；运行中的旧进程不会自动热加载修复，重启 monitor 后才生效。

| 字段 | 旧问题 | 修复 |
|---|---|---|
| `gpu_usage_percent` | powermetrics 输出 `GPU HW active residency`，旧正则只匹配 `GPU active residency` | 改为 `GPU (?:HW )?active residency` |
| `npu_usage_percent` | 旧正则未覆盖 `ANE HW` 前缀 | 改为 `ANE (?:HW )?(?:active residency|duty cycle)` |
| `swap` | macOS 输出 `total = ... used = ... free = ...`，旧正则只匹配 `total: ...` | 改为 `total\s*[:=]` 等 |
| `thermal_pressure` | powermetrics 输出 `Current pressure level: Nominal`，旧正则只匹配 `Thermal pressure:` | 改为同时匹配两者 |

额外注意：

- `npu_usage_percent` 为 `null` 的另一个常见原因是 powermetrics 没有传 `ane_power`；先修 sampler，再判断是否为解析问题。
- 流式 bench 记录里 `prompt_tokens` / `prefill_s` 通常为 `null`，这是客户端流式响应拿不到服务端 usage 所致，不是 monitor 错误；monitor 只按 `completion_tokens` 估算窗口吞吐。
- 当前运行中途如需重启 monitor，会重置该阶段 `metrics.jsonl`；非必要不要中断正式阶段。

## 7. 冒烟测试与 dry-run

### 7.1 六题冒烟 manifest

`--query-source manifest` 下 `--limit-per-dataset` 不会裁剪 manifest，因此不能用：

```bash
# 错误：仍会跑完整 sample_882
--query-source manifest --manifest "$RUN_ROOT/sample_882.jsonl" --limit-per-dataset 1
```

必须生成六条独立清单：

```bash
cat > "$RUN_ROOT/smoke_manifest.jsonl" <<'JSONL'
{"dataset_id":"gpqa","record_index":0}
{"dataset_id":"mmlupro","record_index":0}
{"dataset_id":"math500","record_index":0}
{"dataset_id":"arcc","record_index":0}
{"dataset_id":"gsm8k","record_index":0}
{"dataset_id":"mmlu","record_index":0}
JSONL

wc -l "$RUN_ROOT/smoke_manifest.jsonl"
```

期望 `6`。

### 7.2 六题冒烟命令

在 `3:bench` 窗口：

```bash
cd /Users/jiuwen/AA
source "$RUN_ROOT/run.env"

"$PY" tools/bench_accuracy_concurrency.py \
  --engine "$ENGINE" \
  --query-source manifest \
  --manifest "$RUN_ROOT/smoke_manifest.jsonl" \
  --concurrency 1 \
  --service-max-concurrency 1 \
  --mode both \
  --repeat 1 \
  --warmup 0 \
  --max-output-tokens 2048 \
  --timeout 1800 \
  --cold-load-s "$(cat "$RUN_ROOT/logs/cold_load_s.txt")" \
  --server-pid "$(cat "$SERVER_PID_FILE")" \
  --monitor-disk disk0 \
  --max-rss-gib 55 \
  --max-swap-gib 4 \
  --max-error-rate 0.3 \
  --live-counters-file "$SMOKE_WORKLOAD_FILE" \
  --workload-window-s 15 \
  --out "$RUN_ROOT/smoke" \
  2>&1 | tee "$RUN_ROOT/logs/smoke.log"
```

`--mode both` 只用于接口/计时口径冒烟：六题 × 流式/非流式 = 12 个正式探针请求。正式结果必须把流式和非流式分开跑。

其他模型只替换 `--engine`：

```text
Qwen3.6-35B-A3B → splash-qwen36
Gemma-4-26B-A4B → mlxvlm-gemma
```

### 7.3 冒烟验收

必须同时满足：

- Chat Completions 返回 HTTP 2xx；
- 回答非空；
- `usage` 的 token 字段可用，或明确为 `null`；
- `finish_reason` 可解释；
- 判分结果可追溯；
- 流式 TTFT 可解释，非流式 timing 可解释；
- `$SMOKE_WORKLOAD_FILE` 有新增行；
- monitor 的 `metrics.jsonl` 有新增行；
- 服务日志无 `Engine failed`、OOM、EOF、重复重启、Traceback。

检查命令：

```bash
source "$RUN_ROOT/run.env"
wc -l "$SMOKE_WORKLOAD_FILE"
tail -5 "$SMOKE_WORKLOAD_FILE"
rg -n 'Engine failed|OOM|EOF|restart|Traceback' "$SERVER_LOG" || true
ls -lh "$RUN_ROOT/smoke"
```

冒烟未通过，不运行 dry-run 之后的正式阶段。

### 7.4 manifest dry-run

冒烟通过后、服务仍 ready 时执行：

```bash
cd /Users/jiuwen/AA
source "$RUN_ROOT/run.env"

"$PY" tools/bench_accuracy_concurrency.py \
  --engine "$ENGINE" \
  --query-source manifest \
  --manifest "$RUN_ROOT/sample_882.jsonl" \
  --dry-run \
  2>&1 | tee "$RUN_ROOT/logs/dry_run.log"
```

必须检查日志中的 per-dataset 统计：

```text
gpqa=150, mmlupro=150, math500=150, arcc=150, gsm8k=100, mmlu=182
合计=882
```

dry-run 不请求推理服务，但会加载 evaluator 和数据集。若 dry-run 失败，先修数据/依赖，不进入正式矩阵。

---

## 8. 正式评测：非流式与流式

### 8.1 正式公共参数

正式命令必须显式写出：

```text
--query-source manifest
--manifest "$RUN_ROOT/sample_882.jsonl"
--shard-across-levels
--concurrency 1,2,4,8
--service-max-concurrency 8
--repeat 1
--warmup 1
--max-output-tokens 2048
--timeout 1800
--cold-load-s "$(cat "$RUN_ROOT/logs/cold_load_s.txt")"
--server-pid "$(cat "$SERVER_PID_FILE")"
--monitor-disk disk0
--max-rss-gib 55
--max-swap-gib 4
--max-error-rate 0.3
--workload-window-s 15
```

并显式传入模型 fingerprint（示例为 Qwen3.8）：

```text
--engine-version splash-1.5.0
--weight-format gguf
--quantization Q4_K_M
--draft-model /Users/jiuwen/Models/splash/Qwen3.8-27B-DFlash2
--kv-format int8
--state-format bf16
--model-revision-or-digest <固定 revision 或权重 digest>
```

> 关于 `--max-output-tokens 2048`：Splash 当前默认 reasoning effort 为 `xhigh`，`completion_tokens` 会包含大量 reasoning tokens。最终答案可能只有几个 token，但推理过程经常打满 2048；这是评测耗时长的主要原因，不是异常。本轮正式口径保留 2048，不要为了提速擅自调小。阶段完成后必须检查请求明细里的 `finish_reason`：如果大量为 `length`，说明推理被截断，需要评估是否提高 `max_output_tokens` 或调整 reasoning effort。

其他模型按第 13 节替换；Gemma 的 `weight_format=mlx`、`quantization=MLX-4bit`，不传 Splash 专用 draft/KV/state 值，或在报告中明确为不适用。

### 8.2 非流式正式评测

先在 `2:monitor` 启动 nonstream monitor（见 6.3 阶段 2），再在 `3:bench` 执行：

```bash
cd /Users/jiuwen/AA
source "$RUN_ROOT/run.env"

"$PY" tools/bench_accuracy_concurrency.py \
  --engine "$ENGINE" \
  --engine-version 'splash-1.5.0' \
  --weight-format 'gguf' \
  --quantization "$QUANTIZATION" \
  --draft-model '/Users/jiuwen/Models/splash/Qwen3.8-27B-DFlash2' \
  --kv-format 'int8' \
  --state-format 'bf16' \
  --model-revision-or-digest '<fixed-revision-or-digest>' \
  --query-source manifest \
  --manifest "$RUN_ROOT/sample_882.jsonl" \
  --shard-across-levels \
  --concurrency 1,2,4,8 \
  --service-max-concurrency 8 \
  --mode nonstream \
  --repeat 1 \
  --warmup 1 \
  --max-output-tokens 2048 \
  --temperature 0 \
  --top-p 1 \
  --seed 42 \
  --timeout 1800 \
  --cold-load-s "$(cat "$RUN_ROOT/logs/cold_load_s.txt")" \
  --server-pid "$(cat "$SERVER_PID_FILE")" \
  --monitor-disk disk0 \
  --max-rss-gib 55 \
  --max-swap-gib 4 \
  --max-error-rate 0.3 \
  --live-counters-file "$NONSTREAM_WORKLOAD_FILE" \
  --workload-window-s 15 \
  --out "$RUN_ROOT/accuracy/nonstream" \
  2>&1 | tee "$RUN_ROOT/logs/nonstream.log"
```

bench 完成后，停止 nonstream monitor 并确认收尾完成。

### 8.3 流式正式评测

先在 `2:monitor` 启动 stream monitor（见 6.3 阶段 3），再在 `3:bench` 执行：

```bash
cd /Users/jiuwen/AA
source "$RUN_ROOT/run.env"

"$PY" tools/bench_accuracy_concurrency.py \
  --engine "$ENGINE" \
  --engine-version 'splash-1.5.0' \
  --weight-format 'gguf' \
  --quantization "$QUANTIZATION" \
  --draft-model '/Users/jiuwen/Models/splash/Qwen3.8-27B-DFlash2' \
  --kv-format 'int8' \
  --state-format 'bf16' \
  --model-revision-or-digest '<fixed-revision-or-digest>' \
  --query-source manifest \
  --manifest "$RUN_ROOT/sample_882.jsonl" \
  --shard-across-levels \
  --concurrency 1,2,4,8 \
  --service-max-concurrency 8 \
  --mode stream \
  --repeat 1 \
  --warmup 1 \
  --max-output-tokens 2048 \
  --temperature 0 \
  --top-p 1 \
  --seed 42 \
  --timeout 1800 \
  --cold-load-s "$(cat "$RUN_ROOT/logs/cold_load_s.txt")" \
  --server-pid "$(cat "$SERVER_PID_FILE")" \
  --monitor-disk disk0 \
  --max-rss-gib 55 \
  --max-swap-gib 4 \
  --max-error-rate 0.3 \
  --live-counters-file "$STREAM_WORKLOAD_FILE" \
  --workload-window-s 15 \
  --out "$RUN_ROOT/accuracy/stream" \
  2>&1 | tee "$RUN_ROOT/logs/stream.log"
```

### 8.4 为什么正式模式必须分开

`--mode both` 只适合冒烟，不适合正式结果：

- 同一 stage 同时包含流式和非流式请求，会混合两类时延口径；
- 流式 TTFT 是客户端首内容时间，非流式 TTFT 可能来自服务端 timing；
- 两种模式的 Decode 算法和可用字段不同；
- 报告必须分别汇总、分别出图、分别标注 `timing_source`。

### 8.5 不同模型的命令替换

| 模型 | `--engine` | `--quantization` | 监控目录 | 输出目录 |
|---|---|---|---|---|
| Qwen3.8-27B | `splash-qwen38` | `Q4_K_M` | `qwen38-*` | `.../qwen38/nonstream`、`.../qwen38/stream` |
| Qwen3.6-35B-A3B | `splash-qwen36` | `MLX-4bit` | `qwen36-*` | `.../qwen36/nonstream`、`.../qwen36/stream` |
| Gemma-4-26B-A4B | `mlxvlm-gemma` | `MLX-4bit` | `gemma4-*` | `.../gemma4/nonstream`、`.../gemma4/stream` |

每个模型必须使用新的 `RUN_ROOT`。禁止把不同模型写进同一输出目录，禁止跨引擎合并总排名。

---

## 9. 并发语义、档位与切片

### 9.1 默认并发阶梯

本轮正式采用：

```text
1 → 2 → 4 → 8
```

脚本按给定顺序执行。若 `--service-max-concurrency` 小于某个档位，该档会被跳过并在命令行告警。启动服务时 Splash `--queue-size` / MLX-VLM `--max-num-seqs` 必须 `>= 8`，本轮固定为 `8`。

如需恢复五档满矩阵，再显式使用 `--concurrency 1,2,4,8,16`，并把服务并发上限同步改为 `16`。

### 9.2 每个档位的执行内容

对每个并发档位：

1. 按 `--warmup 1` 执行预热；
2. 对当前档位的完整 query 集合并发请求并判分；
3. 写 `stage_<并发>.json` 和 `stage_<并发>_requests.jsonl`；
4. 汇总内存、swap、SSD、token 吞吐、TTFT、Prefill/Decode、准确率；
5. 检查熔断；
6. 无熔断则进入下一档。

`--warmup 1` 的含义是：每个 mode 先对前 `min(concurrency, len(queries))` 个问题各跑一遍；预热请求不进入准确率和吞吐统计。非流式与流式是两个独立 bench 进程，因此各自会做一次预热。

### 9.3 本轮正式采用：四档跨档切片

为避免 `4 × 882` 的过长耗时，本轮正式流程默认启用：

```bash
--shard-across-levels
--concurrency 1,2,4,8
--service-max-concurrency 8
```

语义：

- query 按并发档数 round-robin 切片；
- 四档合计只跑一遍完整 882 题；
- 每档约 `221 / 221 / 220 / 220` 题；
- 每档准确率不能直接横向比较，只能作为该档观察值。

若使用该模式，报告必须标注：

```text
省时切片模式；各档仅覆盖部分 query，准确率仅作该档观察值，不能直接与其它档比较。
```

### 9.4 可选：每档完整 manifest

只有时间允许且需要“每档都在同一批 882 题上比较准确率”时，才关闭切片：

```bash
--concurrency 1,2,4,8
```

此时每个档位都跑完整 882 题：

```text
concurrency=1 → 882 题
concurrency=2 → 882 题
concurrency=4 → 882 题
concurrency=8 → 882 题
```

优点：每档准确率和性能可直接横向比较。  
代价：总请求量接近 `4 × 882`，预计耗时很长。

### 9.5 没有“每 20 分钟自动升档”

当前脚本没有基于时间的自动升档能力。它只在当前并发档完整结束后进入下一档。

如果必须按时间窗口切换并发，需要用外部编排控制多个独立 bench 进程或增加停止条件；不能把脚本描述成已支持“每 20 分钟自动加一档”。本 SOP 的正式口径是按档跑完再升档。

---

## 10. 熔断、OOM 防护与人工巡检

### 10.1 默认熔断阈值

```text
进程峰值 RSS：55 GiB
swap 增量：4 GiB
错误率：30%
```

对应参数：

```bash
--max-rss-gib 55
--max-swap-gib 4
--max-error-rate 0.3
```

阈值的目的：在 64GB 统一内存机器上，为 macOS、Metal 缓冲和文件缓存留出空间，避免模型逼近整机 OOM。阈值不是“安全许可”，不能替代人工巡检。

### 10.2 熔断的实际行为

`bench_accuracy_concurrency.py` 的熔断发生在**每个 stage 完成之后**：

- 计算该 stage 的 `peak_process_gib`、`swap_delta_gib`、`error_rate`；
- 任一超过阈值则停止进入下一并发档；
- 不会在中途逐秒强制终止当前 stage；
- 不会自动杀掉推理服务。

触发后 `manifest.json` 写入：

```text
aborted_at_concurrency
abort_reasons
stable_max_concurrency
```

若第 1 档就触发，`stable_max_concurrency` 可能为 `null`，不能伪造为 1。

### 10.3 触发后的动作

任一阈值触发：

- 停止升档；
- 保留当前 stage 汇总、请求明细、服务日志和监控曲线；
- 检查 OOM、服务重启、热降频、swap、SSD 读取；
- 把前一档视为稳定最大并发；
- 若服务已崩溃或 RSS 仍快速上升，立即人工停止 bench 和推理服务；
- 不覆盖失败现场。

### 10.4 人工巡检频率与内容

正式运行期间每 `15～30` 分钟巡检一次。若 RSS、swap 或错误率接近阈值，缩短到每 5 分钟。

检查：

- 服务 PID 是否仍存在，PID 是否变化；
- 进程 RSS 是否持续增长；
- swap 是否快速上升；
- thermal pressure 是否明显恶化；
- GPU/ANE 频率和功耗是否异常下降；
- SSD 是否持续高速读取；
- 服务日志是否出现 `Engine failed`、`OOM`、`EOF`、`restart`、`Traceback`；
- 当前 stage 的错误率；
- monitor 的 `metrics.jsonl` 与 workload JSONL 是否持续增长。

巡检命令：

```bash
source "$RUN_ROOT/run.env"
SERVER_PID="$(cat "$SERVER_PID_FILE")"

ps -p "$SERVER_PID" -o pid,rss,%cpu,etime,command
sysctl vm.swapusage
memory_pressure | tail -20
tail -5 "$SERVICE_MONITOR_DIR/metrics.jsonl" 2>/dev/null || true
tail -5 "$NONSTREAM_MONITOR_DIR/metrics.jsonl" 2>/dev/null || true
tail -5 "$STREAM_MONITOR_DIR/metrics.jsonl" 2>/dev/null || true
wc -l "$SMOKE_WORKLOAD_FILE" 2>/dev/null || true
wc -l "$NONSTREAM_WORKLOAD_FILE" 2>/dev/null || true
wc -l "$STREAM_WORKLOAD_FILE" 2>/dev/null || true
rg -n 'Engine failed|OOM|EOF|restart|Traceback' "$SERVER_LOG" || true
```

查看最近 stage 的资源与错误率：

```bash
LATEST_STAGE="$(ls -t "$RUN_ROOT"/accuracy/nonstream/stage_*.json 2>/dev/null | head -1)"
if [ -n "$LATEST_STAGE" ]; then
  "$PY" - "$LATEST_STAGE" <<'PY_STAGE'
import json
import sys
p = sys.argv[1]
s = json.load(open(p, encoding="utf-8"))
print("file", p)
print("concurrency", s.get("concurrency"))
print("memory", s.get("memory"))
print("error_rate", s.get("error_rate"))
print("accuracy", s.get("accuracy"))
print("requests_per_s", s.get("requests_per_s"))
print("decode_tok_per_s_mean", s.get("decode_tok_per_s_mean"))
print("ttft_s_mean", s.get("ttft_s_mean"))
PY_STAGE
fi
```

### 10.5 OOM 风险控制

- 在 16 并发前确认 `--max-memory` / KV / state 配置没有逼近整机内存；
- 先跑 1、2、4、8，再进入 16；不要直接从 1 跳 16；
- 每档结束检查 RSS 和 swap，不只看是否成功返回；
- 若 monitor 观察到 RSS 快速爬升、swap 持续增长、GPU/NPU 数据消失或服务日志出现 OOM，立即人工停止 bench；
- 停止服务后等待端口释放，再决定重跑哪个档位；
- 不为了完成矩阵而关闭熔断。

---

## 11. 产物、验收与收尾

### 11.1 输出目录规范

```text
$RUN_ROOT/
  run.env
  metadata.txt
  baseline/
    preflight.txt
  logs/
    splash-qwen38.log
    splash-qwen38.pid
    service_t0.txt
    ready_t1.txt
    cold_load_s.txt
    ready.json
    status.json
    models.json
    single_request.json
    smoke.log
    dry_run.log
    nonstream.log
    stream.log
    monitor-service-smoke.log
    monitor-nonstream.log
    monitor-stream.log
  monitor/
    powermetrics.txt
    service-smoke/
      manifest.json
      metrics.jsonl
      charts/
    nonstream/
      manifest.json
      metrics.jsonl
      charts/
    stream/
      manifest.json
      metrics.jsonl
      charts/
  sample_882.jsonl
  sample_882.sha256
  smoke_manifest.jsonl
  smoke/
    manifest.json
    summary.md
    stage_1.json
    stage_1_requests.jsonl
  accuracy/
    nonstream/
      manifest.json
      summary.md
      stage_1.json
      stage_1_requests.jsonl
      stage_2.json
      stage_2_requests.jsonl
      stage_4.json
      stage_4_requests.jsonl
      stage_8.json
      stage_8_requests.jsonl
      stage_16.json
      stage_16_requests.jsonl
    stream/
      ...
```

如果 gemma 使用 `gemma4-server.log` / `.pid`，目录内容相应替换，不改变阶段结构。

### 11.2 正式验收清单

- [ ] 固定 manifest 共 `882` 题，六集数量正确；
- [ ] manifest SHA256 已记录且符合 `52208bf7...f2456`；
- [ ] Python >= 3.10；
- [ ] `openai`、`psutil`、`datasets`、`matplotlib` 导入成功；
- [ ] `EvaluatorFactory` 可实例化，`compileall` 通过；
- [ ] 独占窗口已确认，共享推理负载已释放或隔离；
- [ ] `powermetrics` 文件在评测期间持续增长；
- [ ] 三阶段 monitor 都有独立 `metrics.jsonl` 和 `manifest.json`；
- [ ] smoke/nonstream/stream 的 workload 文件没有混用；
- [ ] 冷加载时间来自服务启动到 ready，且注明约 1 秒轮询分辨率；
- [ ] 单请求探活成功并保存响应；
- [ ] 六题冒烟通过；
- [ ] dry-run 总数为 `882`；
- [ ] 非流式与流式分别完成，输出目录分开；
- [ ] 每个并发档位都有 `stage_<并发>.json` 和请求明细；
- [ ] 每档 query 数为 `882`，除非报告明确标注切片模式；
- [ ] 失败请求保留，成功率和错误率单独报告；
- [ ] 缺失字段为 `null`；
- [ ] `finish_reason`、usage、TTFT、Prefill、Decode 口径已说明；
- [ ] 熔断结果与稳定最大并发写入 `manifest.json`；
- [ ] 图表与原始采样成组保存；
- [ ] 不同引擎、量化分表；
- [ ] 报告写清模型、引擎、版本、权重格式、量化、上下文、KV/state、并发、采样参数和硬件。

### 11.3 收尾步骤

1. 等待当前 bench 正常退出；
2. 停止当前阶段 `monitor_machine.py`（`Ctrl-C` 或到达 duration）；
3. 等待 monitor 完成最终 `metrics.jsonl` 补写和 PNG 图表生成；
4. 停止 `powermetrics`（`Ctrl-C`）；
5. 停止推理服务并确认端口释放；
6. 记录服务结束时间和残留进程；
7. 不删除失败样本、错误日志和原始采样；
8. 保存本轮 commit、模型 revision、权重哈希和启动参数。

停止服务：

```bash
source "$RUN_ROOT/run.env"
kill "$(cat "$SERVER_PID_FILE")" || true
sleep 5
ps -p "$(cat "$SERVER_PID_FILE")" || true
lsof -nP -iTCP:"$PORT" -sTCP:LISTEN || true
```

---

## 12. 故障处理

| 现象 | 优先检查 | 处理 |
|---|---|---|
| Python 3.9 类型错误 | `str | Benchmark` 报错 | 切换到 Python >= 3.10，远端的 `.venv-bench` 不得默认使用 |
| 端口被占用 | `lsof -iTCP:<port>` | 等待旧进程退出或更换空闲窗口；不能强杀未知服务 |
| `/ready` 超时 | 服务日志、RSS、swap | 不进入冒烟；保留日志，检查 OOM/EOF/模型路径 |
| `/health` 不是 `model_ready=true` | MLX-VLM 日志、模型目录 | 等待加载完成；加载失败则停止 |
| 探活无回答 | 单请求、`finish_reason`、usage | 修复接口/模型配置，不启动正式评测 |
| `Engine failed` / 重复重启 | 服务日志 | 停止升档，保留现场，重新单独 smoke |
| RSS 超过 55 GiB | stage 汇总 `memory` | 触发熔断，回退前一档；服务异常时人工停止 |
| swap 增量超过 4 GiB | `sysctl vm.swapusage` | 触发熔断，检查内存预算和共享负载 |
| 错误率超过 30% | stage 请求明细 | 触发熔断，检查超时/HTTP 错误/服务崩溃 |
| workload 文件不增长 | `--live-counters-file` 绝对路径、bench 日志 | 修正路径后重跑当前阶段，不能沿用不完整阶段 |
| monitor 无 GPU 数据 | powermetrics 文件、sudo、sampler | 标“未采集”；不得补 0 |
| 图表缺失 | 是否安装 matplotlib | 安装依赖；原始 JSON 仍应保留，不用旧图代替 |
| 准确率异常偏低 | prompt、判分器、样本对齐 | 检查 `(dataset_id, record_index)` 与 evaluator；不要直接发布 |

---

## 13. 模型附录

### 13.1 Qwen3.8-27B

```text
engine preset: splash-qwen38
port: 18080
api model: Qwen/Qwen3.8-27B
quantization: GGUF Q4_K_M
kv: int8
state: bf16
max context: 32K
max memory: 40G
queue size: 16
```

启动命令见 5.3；正式命令见 8.2/8.3。

### 13.2 Qwen3.6-35B-A3B

```text
engine preset: splash-qwen36
port: 18081
api model: mlx-community/Qwen3.6-35B-A3B-4bit
quantization: MLX 4bit affine
kv: int8
state: bf16
max context: 32K
max memory: 42G
queue size: 16
```

启动命令见 5.4。正式命令中把 `--engine-version` 设为 `splash-1.5.0`、`--weight-format` 设为 `mlx`、`--draft-model` 换为 Qwen3.6 的 DFlash2、`--quantization` 换为 `MLX-4bit`。

### 13.3 Gemma-4-26B-A4B

```text
engine preset: mlxvlm-gemma
port: 18091
api model: /Users/jiuwen/Models/gemma-4-26b-a4b-it-4bit
quantization: MLX 4bit affine
server: mlx_vlm.server
max tokens: 2048（单次生成上限，不是上下文）
max num seqs: 16
```

启动命令见 5.5。Gemma 不适合套用 Splash 的 DFlash2/KV/state 参数；报告中明确“不适用”。

### 13.4 Qwen3.8-Flash-Next（第二轮实验）

前置门槛：

- 第一轮表 A/B/C 报告完成；
- 低比特权重已下载并校验哈希；
- 明确这是实验性口径，不与第一轮排名合并。

启动示例：

```bash
/opt/homebrew/bin/splash serve \
  --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 \
  --port 18082 \
  --max-context 32K \
  --max-memory 48G \
  --ple-memory 2G \
  --expert-memory 16G \
  --sparse-kv-memory 1200M \
  --kv-format int8 \
  --state-format bf16 \
  --queue-size 1 \
  --request-timeout 30m
```

第二轮要求：

- 先单请求冒烟，再考虑低并发缩减矩阵；
- 记录 PLE/专家/IQ2 数据是否从 SSD 读取；
- 结果单独成表；
- 与 Qwen3.8/Qwen3.6/Gemma 不合并排名；
- 资产未完成下载和校验时，不得启动正式评测。

---

## 14. 快速执行清单

```text
[ ] ssh macmini-m5-jiuwen && cd /Users/jiuwen/AA
[ ] 确认 Python >= 3.10，openai/psutil/datasets/matplotlib 可用
[ ] 确认独占窗口、端口空闲、共享推理负载已隔离
[ ] 记录机器基线
[ ] 创建新的 RUN_ROOT 与 run.env
[ ] 复制并校验 sample_882.jsonl（sha256 52208bf7...f2456）
[ ] 启动 powermetrics
[ ] 启动 service-smoke monitor
[ ] 记录 T0 并启动服务
[ ] 等待 ready，记录 cold_load_s
[ ] 探活 /ready /status /v1/models /chat/completions
[ ] 六题冒烟（mode=both）
[ ] 882 题 dry-run
[ ] 启动 nonstream monitor，跑非流式 1,2,4,8（--shard-across-levels）
[ ] 启动 stream monitor，跑流式 1,2,4,8（--shard-across-levels）
[ ] 检查熔断、stable_max_concurrency 和产物
[ ] 停止服务、monitor、powermetrics
[ ] 生成分引擎报告，不合并排名
```
