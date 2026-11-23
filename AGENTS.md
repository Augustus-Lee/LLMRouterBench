# AGENTS.md — LLMRouterBench 跨会话记忆

> 本文件是 Codex 在本仓库工作的长期上下文入口。新会话先读这里，再按需读 `README_zh.md` 和 `docs/`。
> 最后核对：2026-11-23（Asia/Shanghai）。状态会变化，涉及远端机器、模型服务、未提交文件时先重新核实。

## 1. 用户与仓库约定

- 用户主要使用中文；默认用中文沟通。
- 所有需要保存到本地的文档默认使用 Markdown（`.md`），放在当前工作目录的合适位置（如 `docs/`）；用户指定其他格式时除外。
- 不记录、回显或提交密码、私钥、API Key、token。配置里只使用环境变量名。
- 本仓库路径：`/Users/augustus/Desktop/LLMBenchmark`
- 上游项目：`https://github.com/meijamke/LLMRouterBench.git`（`upstream`）
- 用户 fork：`https://github.com/Augustus-Lee/LLMRouterBench.git`（`origin`）
- 当前分支：`main`，跟踪 `origin/main`
- 本次核对时 HEAD：`af344b7 route: 端侧大模型 9B→14B，LLM分类改用prompt输入，修复data_loader UTF-8读取`
- 代码修改默认从最新 `origin/main` 开 `codex/` 前缀分支；不要直接向上游 `upstream` 推送。
- 提交前先看 `git status`：工作区可能已有用户未提交的文档，不能擅自删除、覆盖或混入无关提交。

## 2. 项目是什么

`LLMRouterBench` 是“LLM 路由”研究平台，不是普通的单模型 benchmark：

1. Collector 让同一批题目经过多个模型，生成“题目 × 模型”的答案矩阵；
2. Evaluator 对每个答案判分；
3. Adaptor 把统一结果转成各路由算法输入；
4. Router 为每题选择模型；
5. 离线查表回放，比较准确率、token、成本与 Pareto 前沿。

核心公式：

```text
质量 = 被路由模型在该题上的得分
成本 = 分类器/路由器开销 + 被路由模型的 token 成本
```

路由评估不需要在回放时重新调用回答模型；因此完整答案矩阵是最重要的中间资产。

官方框架规模约为 33 个模型、21+ 数据集、10 种路由算法。**本地二次开发重点不是复刻全部官方实验，而是端侧/云端路由。**

## 3. 本地二次开发主线

### 3.1 六个数据集与样本口径

本地端侧实验常用 6 个数据集：

| dataset_id | 数据文件 | 常用样本数 | 判分方式 |
|---|---|---:|---|
| `gpqa` | `data/GPQA/gpqa_diamond.json` | 198（历史矩阵用 150） | 选项匹配 |
| `mmlupro` | `data/MMLUPro/test.json` | 1001（历史矩阵用 150） | 选项匹配 |
| `math500` | `data/MATH500/test.json` | 500（历史矩阵用 150） | 答案抽取 + 数学判分 |
| `arcc` | `data/arc_c/arc_test.json` | 1172（历史矩阵用 150） | 选项匹配 |
| `gsm8k` | `data/GSM8K/test.json` | 100 | 数值匹配 |
| `mmlu` | `data/MMLU/test.json` | 182 | 选项匹配 |

历史 XRouter 回放矩阵是 **882 题**：`gpqa/mmlupro/math500/arcc` 各 150，`gsm8k` 100，`mmlu` 182。
不要把它和当前全量数据条数混用。新增回放必须按 `(dataset_id, record_index)` 严格对齐。

### 3.2 当前 5 模型答案池

当前端侧分析代码中的模型池：

- 端侧候选：`Qwen3.5-4B`、`Qwen3-8B`、`Qwen3-14B`、`Qwen3.5-9B`
- 云端：`DeepSeek-V4.1-Flash`

端侧模型的离线回放成本按 **0** 计（电力、折旧等暂不计）；云侧按硅基流动真实价格计：

```text
DeepSeek-V4.1-Flash: 输入 ¥2.00 / M tokens，输出 ¥8.00 / M tokens
```

成本口径是用户明确修正过的，不要改回 API 代跑价。`Qwen3.5-4B/8B/9B/14B` 在回放中均视为端侧 0 成本。

### 3.3 XRouter 三角色语义

```text
提问
  → 端侧 4B 做 5 档难度分类
       SIMPLE / MEDIUM / COMPLEX / RESEARCH / REASONING
  → tier <= localCapabilityLevel → 端侧大模型
  → tier > localCapabilityLevel  → 云端 DeepSeek
```

- LLM 分类失败或解析失败时回退 `classify_heuristic`；语义是“故障只降不升”。
- 分类输入统一使用 `record.prompt`（完整题干和选项），**不要改回 `origin_query`**；MCQ 的 `origin_query` 可能缺选项并低估难度。
- 当前 HEAD 的 `tools/eval_3model_full.py` 将端侧回答模型设为 `Qwen3-14B`，默认能力阈值 `MEDIUM`。
- 历史报告 `docs/edge_router_3model_eval_report.md` 的主体仍是 **9B 时代**结果，和 HEAD 代码并不完全同步；引用数值前必须重新运行并更新报告。

### 3.4 已实现的关键能力

- 端侧 4 模型 + 云侧 DeepSeek 的 882 题答案矩阵与离线回放；
- LLM/启发式两套难度分类；
- `SIMPLE` 到 `NONE` 的能力阈值扫描与质量-成本 Pareto；
- 单模型、AVG、Random、MaxExpert、Category Expert、Oracle 基线；
- MMLU/MMLU-Pro 的 `category` 字段贯通，支持 subject 粒度 Category Expert；
- 端侧成本=0 的统一重算；
- 分类器限流处理：长超时、并发控制和退避重试。

## 4. 重要代码与配置入口

```text
data_collector/cli.py                 Collector CLI
data_collector/config_loader.py       配置加载
data_collector/planner.py             dataset × split × model 运行计划
data_collector/runner.py              单次采集、判分、落盘
evaluation/factory.py                 Evaluator 注册表
baselines/data_loader.py              读取 results/bench，构建 BaselineRecord
baselines/schema.py                   BaselineRecord / AggregatedStats
baselines/aggregators.py              AVG/Random/MaxExpert/CategoryExpert/Oracle
baselines/device_llm_router.py        XRouter 分类、档位与启发式逻辑
baselines/adaptors/                   各路由算法的数据适配
```

端侧专用配置与脚本：

```text
config/edge_router_config.yaml        4 端侧模型 + DeepSeek，6 数据集
config/edge_router_3model.yaml        仅补采 Qwen3.5-9B
config/baseline_config_edge.yaml      端侧答案矩阵加载配置
tools/collect_9b_882.py               按旧矩阵索引补采 9B
tools/run_xrouter_classifier.py       4B 难度分类并写 tiers CSV
tools/eval_device_llm_router_3model.py  两模型三角色回放
tools/eval_3model_full.py             5 模型全基线 + 3 张图
tools/eval_3model_sweep.py            能力阈值扫描 + Pareto
tools/verify_category_expert.py       Category Expert 独立交叉验证
tools/run_edge_analysis.py            一键端侧分析入口
```

主要输出（多数被 `.gitignore`）：

```text
results/bench/                        原始采集结果矩阵，当前工作区缺失
results/xrouter_classifier_tiers.csv  4B 分类档位（已提交）
results/devllmr_3model_pareto.png      Pareto 图（已提交）
results/xrouter_comparison.png        启发式/LLM/基线对比图（已提交）
results/xrouter_heur_classification.png
results/xrouter_llm_classification.png
```

## 5. 常用命令

```bash
# 环境
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 数据（仓库用 Git LFS）
git lfs install
git lfs pull

# 采集器
python -m data_collector.cli list
python -m data_collector.cli run config/edge_router_config_demo.yaml -y
python -m data_collector.cli run config/edge_router_3model.yaml -y

# XRouter 回放（需要 results/bench 和 tiers CSV）
python tools/run_xrouter_classifier.py
python tools/run_xrouter_classifier.py --retry-failed --concurrency 6
python tools/eval_3model_full.py
python tools/eval_3model_sweep.py

# 轻量静态检查
python -m compileall -q baselines data_collector evaluation tools
```

需要的环境变量名（只写名字，不记录值）：

```text
SILICONFLOW_API_KEY       端侧实验的硅基流动 API Key
OPENROUTER_API_KEY
HUOSHAN_API_KEY
OPENAI_API_KEY
GRADER_API_KEY / GRADER_BASE_URL / GRADER_MODEL_NAME
ARENA_GRADER_API_KEY / ARENA_GRADER_BASE_URL / ARENA_GRADER_MODEL_NAME
MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD
```

## 6. 三台 Mac mini 环境

三台机器统一为 64GB 统一内存：

| SSH alias | IP | 用户 | 硬件 | 角色 |
|---|---|---|---|---|
| `macmini` | `192.168.3.13` | `synco` | M5 Pro 64GB | 已运行 Qwen3.8-27B + 35B-A3B，重点评测机 |
| `macmini-m5-jiuwen` | `192.168.3.5` | `jiuwen` | M5 Pro 64GB | Qwen3.8-27B、35B-A3B 基线 |
| `macmini-m4-admin` | `192.168.3.7` | `admin` | M4 Pro 64GB | Gemma-4-26B-A4B、35B-A3B、M4 对照 |

已验证：

- 三台均已配置免密 SSH；私钥只在本机 `~/.ssh/id_ed25519`，不要复制进仓库。
- 3.13 当前有 Splash `127.0.0.1:18080` 服务 Qwen3.8-27B GGUF Q4_K_M；
- 3.13 有 Ollama `127.0.0.1:11435` 服务 `qwen3.6:35b-a3b-q4_K_M`；
- NVIDIA PAIR 是聚合/代理层，**不能仅凭 PAIR 列出的 model inventory 判断权重已安装**；
- 3.13 磁盘紧张（约 63GiB 可用），Splash 常驻内存约 22GiB；测别的模型前先错峰/释放。

长期结论：

- 20B～35B 级模型用 4bit 在这三台机器上可行；
- `Qwen3.8-Flash-125B-A6B` 的 Q4 约 65～75GB，不适合作为 64GB 单机稳定基线；
- 125B 只能做 IQ2/Q2、4k～8k、单并发实验，并明确标为实验性；
- 不同推理引擎（MLX-LM、Splash、Ollama、oMLX、llama.cpp、LM Studio）的 TTFT/Decode 不可直接横比；
- 正式性能结果必须记录模型仓库/revision、量化、实际文件大小、引擎版本、上下文、KV cache、并发、预热、采样参数、峰值内存、swap。

优先阅读：

```text
docs/三台MacMini_配置与模型部署可行性评估.md   # 三台机器的权威汇总
docs/MacMini_已装模型清单.md                  # 3.13 的详细服务与模型快照
docs/MAC_文本模型性能评测方案.md              # Mac 性能评测方案
docs/SSH_连接说明.md                          # 3.13 的连接与工作流
```

注意：`docs/MacMini_已装模型清单.md` 内有一处旧链接写成了不存在的 `两台MacMini_...`，实际文件是 `三台MacMini_...`。

## 7. 仓库与数据风险

1. **`results/bench/` 当前不存在**。它是 `.gitignore` 排除的大体积采集结果，许多回放脚本没有它就不能运行。不要编造或把空目录当成完整矩阵。
2. `data/**` 由 Git LFS 管理。看到 130 字节左右的 JSON/GGUF 可能是 LFS pointer，不是有效数据；先查文件大小。
3. `docs/` 下有用户创建但尚未提交的 Markdown；这些内容有价值，编辑前先读，不要用生成内容覆盖。
4. `results/xrouter_classifier_tiers.csv` 已提交，但它只代表某次分类器运行；换模型、换 prompt 口径或修分类器后要重新生成。
5. 端侧报告与 HEAD 代码存在 9B→14B 的版本差异；以代码和当前重跑结果为准，不直接引用历史表格。
6. 仓库没有正式单元测试套件；`tools/test_embedding_model.py` 是工具，不是完整测试。修改后至少做 `compileall`，涉及回放时再跑端到端脚本并核对样本数、成本和降级数。
7. `results/devllmr_3model_pareto.png` 等图片可能对应旧口径；图表与 CSV 应成组更新，避免图和数不一致。

## 8. 当前任务状态与待办

截至 2026-11-23：

- XRouter 三角色离线回放已打通；
- 4B 限流问题已通过 300s 超时、并发 6、退避重试解决，历史矩阵达到 882/882 LLM 打档；
- HEAD 已将端侧回答模型切到 `Qwen3-14B`，分类输入改为 `prompt`，并修复 `data_loader` 的 UTF-8 读取；
- 端侧成本=0、云侧真实计费口径已固定；
- **仍待完成**：用 14B 口径重跑并刷新 `docs/edge_router_3model_eval_report.md` 与相关图表；确认报告中的“9B only/最终路由”叙述全部切换或在版本说明中明确保留历史结果。

Mac 侧后续顺序：

1. 在 3.13 跑 Qwen3.8-27B Splash 基线（8k/32k/64k）；
2. 释放 Splash 后测 Qwen3.6-35B-A3B；
3. 安装 Gemma-4-26B-A4B 4bit 并错峰测试；
4. 125B-A6B 暂不进入正式基线；
5. 将 TTFT、Prefill、Decode、峰值内存、swap、失败率汇总为 Markdown。

## 9. 工作方式与验证要求

- 先读相关代码和文档，再改代码；不要只根据 README 推断行为。
- 路由/成本类改动必须保持 `(dataset_id, record_index)` 对齐；样本缺失时显式报告，不要静默 inner join。
- 任何 baseline 变更都要确认是否排除 reference models，避免污染 AVG、Random、MaxExpert、Category Expert、Oracle。
- 成本单位必须写明（人民币/美元、每百万 token、输入/输出），并区分分类器成本和回答模型成本。
- 结果声明必须可复现：记录命令、配置、模型版本、数据规模、随机种子和输出路径。
- 交付前运行最小验证；无法运行时说明缺少哪个输入/凭据/远端服务，不要声称已验证。
- 文档结论优先链接到 `docs/` 中的详细说明，本文件只保留跨会话必须知道的内容。
