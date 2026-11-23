# LLMRouterBench 项目详解

> 仓库：`meijamke/LLMRouterBench`
> 本地路径：`/Users/augustus/Desktop/LLMBenchmark`
> 当前 HEAD：`af344b7 route: 端侧大模型 9B→14B，LLM分类改用prompt输入，修复data_loader UTF-8读取`

---

## 一、先给结论：这个项目到底是什么

`meijamke/LLMRouterBench` 不是传统意义上的“选一个模型跑一个 benchmark”，而是一个 **LLM 路由研究平台**：

> 先让同一批题目分别经过多个模型，形成“题目 × 模型”的完整答案矩阵；然后训练一个路由器，为每道题选择一个模型；最后通过离线查表计算整体准确率和成本。

它同时覆盖两类问题：

1. **纯性能路由**
   - 模型池主要是约 7B 的轻量模型。
   - 目标是在多个小模型之间选择，提高平均准确率。
   - 常见指标包括 `AvgAcc`、`Gain@R`、`Gain@B`、`Gap@O`。

2. **性能-成本路由**
   - 模型池包含旗舰模型，调用成本不同。
   - 目标是在准确率和 API 成本之间取得平衡。
   - 常见指标包括 `PerfGain`、`CostSave`、`Pareto`、`ParetoDist`。

项目官方规模是 33 个模型、21+ 数据集、10 种路由算法，约 40 万实例和 18 亿 tokens。

---

## 二、理解项目的关键：“答案矩阵”

整个系统最核心的中间产物是一个二维矩阵：

| 数据集 | 题目索引 | Qwen3-4B | Qwen3-8B | Qwen3-14B | DeepSeek |
|---|---|---:|---:|---:|---:|
| GPQA | 1 | 0/1、tokens、cost | 0/1、tokens、cost | 0/1、tokens、cost | 0/1、tokens、cost |
| MATH500 | 1 | 0/1、tokens、cost | 0/1、tokens、cost | 0/1、tokens、cost | 0/1、tokens、cost |
| MMLU | 1 | 0/1、tokens、cost | 0/1、tokens、cost | 0/1、tokens、cost | 0/1、tokens、cost |

矩阵中的每一行是一道题，每一列是一个模型，单元格里保存：

- 模型答案
- 原始输出
- 正确性分数
- prompt tokens
- completion tokens
- 调用成本
- 附加字段，例如 MMLU 的 subject/category

路由算法本身只输出：

```text
第 1 题 → Qwen3-4B
第 2 题 → DeepSeek
第 3 题 → Qwen3-14B
```

它不直接生成答案，也不直接调用模型。选定模型后，系统回到答案矩阵中查表，得到该模型的分数、token 和成本。

因此，一旦答案矩阵采集完成，就可以：

- 离线比较不同路由算法；
- 扫描不同路由阈值；
- 绘制准确率-成本 Pareto 曲线；
- 避免反复调用昂贵的回答模型。

这也是这个项目适合做路由研究的原因。

---

## 三、完整流水线

主流程可以概括为：

```text
配置加载
  → 生成运行计划
  → 逐模型、逐数据集调用 API
  → Evaluator 判分
  → 标准 JSON 落盘
  → BaselineDataLoader 构建答案矩阵
  → BaselineAggregator 计算单模型和静态基线
  → Adaptor 转换数据
  → 训练路由算法
  → 测试集离线回放
  → 性能、成本、Pareto 分析
```

### 3.1 配置与运行计划

入口文件：

```text
/Users/augustus/Desktop/LLMBenchmark/data_collector/cli.py
```

主要调用链：

```text
ConfigLoader.load()
  → RunPlanner.generate_run_plan()
  → BenchmarkRunner.run_all()
```

其中 Planner 会枚举：

```text
dataset × split × model
```

并决定哪些任务需要运行：

- 非 demo 的已有结果默认跳过；
- 配置 fingerprint 变化后才会重跑；
- 显式设置 `overwrite: true` 才会强制覆盖；
- demo 结果默认不会被正式分析加载。

相关代码：

```text
/Users/augustus/Desktop/LLMBenchmark/data_collector/config_loader.py
/Users/augustus/Desktop/LLMBenchmark/data_collector/planner.py
```

### 3.2 单次评测执行

核心文件：

```text
/Users/augustus/Desktop/LLMBenchmark/data_collector/runner.py
```

执行逻辑是：

1. `Evaluator.load_data(split)` 加载测试数据；
2. 如果是 demo 模式，只截取前 N 条；
3. 创建 `DirectGenerator`；
4. 使用线程池逐题请求模型；
5. 调用 `Evaluator.evaluate()` 判分；
6. 汇总 accuracy、tokens、cost；
7. 调用 `ResultsStorage.save_result()` 落盘。

结果文件路径为：

```text
{output_dir}/bench/{dataset}/{split}/{model}/{dataset}-{split}-{model}-{timestamp}.json
```

单条结果通常包含：

```text
index
origin_query
prompt
prediction
raw_output
ground_truth
score
prompt_tokens
completion_tokens
cost
extra_fields
```

需要注意的实现细节：

> `runner.py` 中实际执行了 `index = index + 1`，所以落盘索引是 1-based；但 `BaselineRecord` 的注释可能描述为 0-based。

自定义答案矩阵对齐时，应按：

```text
(dataset_id, record_index)
```

建立键，不要假定索引一定从 0 或 1 开始。

### 3.3 Generator：真正调用模型的地方

关键文件：

```text
/Users/augustus/Desktop/LLMBenchmark/generators/generator.py
```

实现特征：

- 使用 OpenAI Python SDK；
- 要求服务兼容 `/v1/chat/completions`；
- 请求参数包括 model、messages、temperature、top_p、timeout；
- 支持 `reasoning_effort`；
- `extra_body` 会原样传给 OpenAI SDK；
- 最多重试约 10 次，指数退避；
- 如果响应包含 `usage.cost`，优先使用它；
- 否则按每百万 token 价格计算成本。

成本公式大致是：

```text
prompt_tokens / 1e6 * prompt_price_per_million
+ completion_tokens / 1e6 * completion_price_per_million
```

一个非常重要的问题是：

> 所有重试都失败后，Generator 不抛异常，而是返回字符串 `Generation failed: ...`，token 和 cost 记为 0，通常最终得 0 分。

所以采集完成后必须检查：

```text
Generation failed 出现了多少次
```

否则可能把 API 故障或限流误认为模型能力差。

另外，当前 Generator 没有统一的 `max_tokens` 或 `max_completion_tokens` 一等配置字段，主要依赖服务端默认值。对于推理模型，可能需要在 `extra_body` 中设置输出上限，并先用小样本验证 API 是否接受。

### 3.4 Evaluator：怎么判分

目录：

```text
/Users/augustus/Desktop/LLMBenchmark/evaluation/
```

工厂入口：

```text
/Users/augustus/Desktop/LLMBenchmark/evaluation/factory.py
```

基类：

```text
/Users/augustus/Desktop/LLMBenchmark/evaluation/base_evaluator.py
```

主要判分类型包括：

- 数学题：答案抽取，再用 mathd、SymPy 等比较；
- 选择题：正则抽取 `Answer: LETTER`；
- 代码题：执行单元测试；
- 开放题：LLM-as-Judge 或 rubric 判分。

LLM Judge 默认配置是内网地址和模型：

```text
http://172.30.4.29:8000/v1
Qwen2.5-72B-Instruct
```

使用 HLE、SimpleQA、FrontierScience、SGIBench 等 LLM Judge 数据集前，必须设置：

```bash
export GRADER_BASE_URL="..."
export GRADER_API_KEY="..."
export GRADER_MODEL_NAME="..."
```

否则这些数据集在外部环境通常无法判分。

---

## 四、Baseline 数据加载、聚合和指标

关键文件：

```text
/Users/augustus/Desktop/LLMBenchmark/baselines/schema.py
/Users/augustus/Desktop/LLMBenchmark/baselines/data_loader.py
/Users/augustus/Desktop/LLMBenchmark/baselines/aggregators.py
```

### 4.1 DataLoader

`BaselineDataLoader` 会扫描结果目录，把 JSON 转换成统一的 `BaselineRecord`。

两个容易踩坑的行为：

1. 同一个 dataset/split/model 只保留时间戳最新的文件；
2. `demo=true` 的结果默认跳过。

也就是说，同一模型重复运行时，不是把多次结果合并，而是使用最新一次结果。

### 4.2 主要基线

`BaselineAggregator` 支持以下基线：

- Single Model：单个模型；
- AVG：模型池平均表现；
- Random Router：随机选模型；
- Max Expert：先按 dataset 选平均表现最好的模型；
- Category Expert：先按 `(dataset, category)` 选表现最好的模型；
- Oracle：每道题只要有一个模型答对就算答对。

成本口径也要注意：

- Max Expert 使用被选中专家模型的成本；
- Category Expert 同理；
- Oracle 如果有模型答对，选其中成本最低的答对模型；如果全错，则选成本最低的模型。

### 4.3 哪些是“可部署结果”，哪些只是上界

这一点非常重要：

- Single Model、AVG、Random 属于可部署或近似可部署结果；
- Max Expert 是使用测试集标签按 dataset 事后选择最佳模型；
- Category Expert 是使用测试集标签按 subject 事后选择最佳模型；
- Oracle 是每道题事后知道哪个模型答对。

因此：

> Max Expert、Category Expert、Oracle 都不是现实中的在线路由结果，而是 hindsight 上界。

尤其是 Category Expert，它使用的是测试标签信息。不能把“84% 准确率”直接理解成线上系统能达到的结果。

### 4.4 Dataset-Avg 与 Sample-Avg

项目中的两个平均口径不同：

- `Dataset-Avg`：先计算每个数据集的指标，再对数据集等权平均；
- `Sample-Avg`：合并所有样本后等权平均。

当各数据集题目数量差异较大时，两者会明显不同。写报告时必须明确使用哪种口径。

### 4.5 训练/测试划分

核心原则是先按 prompt 分组，再划分 train/test，而不是直接随机切分记录。

对应函数通常是：

```text
split_by_dataset_then_prompt()
```

这样可以保证：

> 同一道题在不同模型上的输出始终落在同一侧，避免训练集和测试集泄漏。

常见配置是：

```text
train_ratio = 0.8
seed = 42
```

---

## 五、路由算法和 Adaptor

官方 baseline 目录：

```text
/Users/augustus/Desktop/LLMBenchmark/baselines/
```

Adaptor 目录：

```text
/Users/augustus/Desktop/LLMBenchmark/baselines/adaptors/
```

官方覆盖的算法包括：

- RouterDC
- EmbedLLM
- MODEL-SAT
- Avengers
- Avengers-Pro
- HybridLLM / Best-route-llm
- FrugalGPT
- RouteLLM
- GraphRouter
- OpenRouter 对照

Adaptor 的作用是把统一的答案矩阵转换成某个算法需要的数据格式，然后执行训练和测试。

例如 Avengers-Pro 的命令形式是：

```bash
cd /Users/augustus/Desktop/LLMBenchmark

python -m baselines.adaptors.avengerspro_adaptor \
  --config config/baseline_my_eval.yaml \
  --seed 42 \
  --split-ratio 0.8 \
  --output-dir baselines/AvengersPro/data/my_eval
```

要注意不同 baseline 有各自的依赖和训练要求，很多算法需要 PyTorch、Transformers 或 embedding 模型。应使用独立虚拟环境，不建议把全部 `requirements.txt` 混在一个环境中，否则容易出现依赖冲突。

---

## 六、同事二次开发具体增加了什么

重点文档：

```text
/Users/augustus/Desktop/LLMBenchmark/docs/pipeline-and-edge-commits.md
/Users/augustus/Desktop/LLMBenchmark/docs/edge_router_3model_eval_report.md
/Users/augustus/Desktop/LLMBenchmark/docs/task2_minipc_real_device_plan.md
```

### 6.1 接入 SiliconFlow 和端侧模型

关键配置：

```text
/Users/augustus/Desktop/LLMBenchmark/config/edge_router_config.yaml
/Users/augustus/Desktop/LLMBenchmark/config/edge_router_config_demo.yaml
/Users/augustus/Desktop/LLMBenchmark/config/baseline_config_edge.yaml
```

回答池包括：

- Qwen3.5-4B
- Qwen3-8B
- Qwen3-14B
- DeepSeek-V4.1-Flash

数据集包括：

- GPQA
- MMLU-Pro
- MATH500
- ARC-C
- GSM8K
- MMLU

demo 配置使用：

```text
demo_limit: 6
overwrite: true
output_dir: results_demo
```

适合做小规模接口联通和判分冒烟测试。

### 6.2 新增 MMLU 和 GSM8K Evaluator

新增文件：

```text
/Users/augustus/Desktop/LLMBenchmark/evaluation/MMLU/mmlu.py
/Users/augustus/Desktop/LLMBenchmark/evaluation/GSM8K/gsm8k.py
```

并注册到：

```text
/Users/augustus/Desktop/LLMBenchmark/evaluation/factory.py
```

MMLU 和 MMLU-Pro 的 `evaluate()` 会返回 `category`，Runner 将其放入 `extra_fields.category`，DataLoader 再转换成 `BaselineRecord.category`。

这使得 Category Expert 可以按 MMLU subject 或数据集 category 粒度选择模型。

### 6.3 XRouter / DeviceLLMRouter

新增文件：

```text
/Users/augustus/Desktop/LLMBenchmark/baselines/device_llm_router.py
```

难度档位是：

```text
SIMPLE < MEDIUM < COMPLEX < RESEARCH < REASONING
```

核心规则：

```text
题目难度 <= 本地模型能力档位 → 本地模型
题目难度 > 本地模型能力档位  → 云侧模型
```

默认本地能力档位是 `MEDIUM`。

相关分析和回放工具包括：

```text
/Users/augustus/Desktop/LLMBenchmark/tools/eval_device_llm_router_3model.py
/Users/augustus/Desktop/LLMBenchmark/tools/eval_3model_full.py
/Users/augustus/Desktop/LLMBenchmark/tools/eval_3model_sweep.py
/Users/augustus/Desktop/LLMBenchmark/tools/eval_3model_charts.py
/Users/augustus/Desktop/LLMBenchmark/tools/run_xrouter_classifier.py
```

这些工具主要做：

- 离线回放；
- 本地能力档位扫描；
- 准确率-成本曲线；
- Pareto 分析；
- 分类器驱动的路由实验。

### 6.4 同事报告中的参考数字

报告口径是“4B 分类器 + 9B 端侧 + DeepSeek 云侧”，端侧成本暂定为 0：

| 方案 | 准确率 | 成本 |
|---|---:|---:|
| X-Router，4B LLM 分类器 | 58.62% | ¥3.245 |
| X-Router，启发式分类器 | 46.37% | ¥0.1095 |
| 9B only | 45.92% | ¥0 |
| 4B only | 61.11% | ¥0 |
| 8B only | 56.46% | ¥0 |
| 14B only | 58.50% | ¥0 |
| DeepSeek only | 82.20% | ¥4.743 |
| Max Expert | 84.13% | ¥3.708 |
| Category Expert | 84.35% | ¥3.583 |
| Oracle | 93.99% | ¥0.967 |

这里有三个必须强调的点：

1. 这些数字是 **9B 版本报告口径**，不是最新 14B 完整评测结果；
2. Max Expert、Category Expert、Oracle 是事后上界；
3. 端侧成本按 0 计，没有计算电力、能耗和设备折旧；DeepSeek 使用的是人民币价格：输入 2 元/百万 token，输出 8 元/百万 token。

另外，表中 4B-only 的 61.11% 实际高于 4B LLM 分类器 X-Router 的 58.62%。因此不能简单说“LLM 分类器整体更好”，更准确的说法是：

> 在当前成本口径和分类效果下，4B LLM 分类器显著优于启发式分类器，但相比 4B-only 的准确率优势并不存在，需要结合更多云侧调用和成本约束解释。

官方 baseline 配置中的成本多为美元口径，与同事的人民币口径不能直接混用。

---

## 七、当前仓库存在一个明确的 9B/14B 版本不一致

最新提交标题说“端侧大模型 9B→14B”，但代码并未全部同步：

- `/Users/augustus/Desktop/LLMBenchmark/tools/eval_3model_full.py` 使用 `Qwen3-14B`；
- `/Users/augustus/Desktop/LLMBenchmark/tools/eval_device_llm_router_3model.py` 仍使用 `Qwen3.5-9B`；
- `/Users/augustus/Desktop/LLMBenchmark/config/edge_router_3model.yaml` 仍描述 9B；
- `/Users/augustus/Desktop/LLMBenchmark/docs/edge_router_3model_eval_report.md` 主要是 9B 口径。

最新提交实际改动主要集中在：

```text
baselines/data_loader.py
results/xrouter_classifier_tiers.csv
三张结果图
tools/eval_3model_full.py
```

因此复现前必须先统一确认：

> 端侧回答模型到底是 Qwen3.5-9B 还是 Qwen3-14B？

否则会出现“配置写 9B、完整评估脚本跑 14B、报告解释 9B”的混合口径。

---

## 八、基于该项目评测其他大模型的操作建议

### 第一步：明确评测目标

先决定属于哪一种：

1. **只测单模型能力**
   - 用 Collector 采集；
   - 用 Aggregator 输出单模型准确率和成本。

2. **构造多模型路由池**
   - 所有候选模型必须跑完全相同的数据集和 split；
   - 缺失模型的题目通常不能进入完整对齐矩阵。

3. **做纯性能路由**
   - 关注 AvgAcc 和相对最佳单模型的增益。

4. **做性能-成本路由**
   - 必须正确填写输入/输出价格；
   - 统一币种；
   - 决定是否计算本地推理成本。

### 第二步：解决当前环境前置条件

当前机器上的主要问题是数据还没有真正拉取。

已确认：

```text
data 下共 2016 个文件，其中 2011 个是 Git LFS pointer
```

例如：

```text
/Users/augustus/Desktop/LLMBenchmark/data/MATH500/test.json
```

目前只是 LFS pointer 文本，不是真正的 JSON 数据。当前环境也没有 `git lfs` 命令。

建议先执行：

```bash
brew install git-lfs
git lfs install

cd /Users/augustus/Desktop/LLMBenchmark
git lfs pull
```

如果 LFS 拉取不稳定，也可以从 Hugging Face 获取：

```text
https://huggingface.co/datasets/NPULH/LLMRouterBench
```

另外，当前 `results/` 中没有完整的 `results/bench`，所以暂时无法复现官方论文指标或同事报告。可以从这些位置获取发布包：

- `/Users/augustus/Desktop/LLMBenchmark/results/download.md`
- 官方 Google Drive
- 百度网盘 `bench-release.tar.gz`，提取码 `mmbf`
- Hugging Face 数据集

### 第三步：创建独立 Python 环境

当前系统 Python 是 3.13.5。建议：

```bash
cd /Users/augustus/Desktop/LLMBenchmark

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

如果准备运行 RouterDC、EmbedLLM、MODEL-SAT 等需要 PyTorch 的算法，建议优先使用 Python 3.10 或 3.11，并为不同算法建立独立环境。主 Collector 环境和各 baseline 环境最好分开。

### 第四步：给新模型创建独立配置

不要直接改官方配置。建议新建：

```text
/Users/augustus/Desktop/LLMBenchmark/config/eval_my_model.yaml
```

参考模板：

```yaml
models:
  - name: MyModel
    api_model_name: org/MyModel
    base_url: https://api.siliconflow.cn/v1
    api_key: SILICONFLOW_API_KEY
    generator_type: direct
    temperature: 0
    timeout: 1800
    extra_body:
      enable_thinking: false
    pricing:
      prompt_price_per_million: 0.5
      completion_price_per_million: 2.0

datasets:
  - dataset_id: gpqa
    splits: [test]
  - dataset_id: math500
    splits: [test]

run:
  kind: bench
  output_dir: results_my_eval
  overwrite: false
  concurrency: 8
  log_level: INFO
  demo_mode: false
  demo_limit: 10
```

实际字段应结合仓库中的 `config_loader.py` 和已有配置确认。不要将真实 API Key 写入 Git 仓库，使用环境变量管理。

### 第五步：先做 demo 冒烟

建议复制一份 demo 配置，只保留一个模型、一两个数据集，并使用：

```yaml
run:
  demo_mode: true
  demo_limit: 6
  overwrite: true
  output_dir: results_my_eval_demo
```

然后先查看运行计划：

```bash
cd /Users/augustus/Desktop/LLMBenchmark

python -m data_collector.cli info config/eval_my_model.demo.yaml
```

再执行：

```bash
python -m data_collector.cli run config/eval_my_model.demo.yaml -y
```

这一步的目的不是看最终分数，而是检查：

- API 配置是否可用；
- 模型名是否正确；
- 请求参数是否被服务端接受；
- 输出是否被截断；
- Token 和 cost 是否正确；
- 判分器能否处理模型输出格式；
- 是否出现 `Generation failed`；
- 是否出现空响应或异常 JSON。

### 第六步：检查结果 JSON

重点检查生成文件中的：

```text
prediction
raw_output
score
prompt_tokens
completion_tokens
cost
extra_fields
```

统计失败数量：

```text
raw_output 是否包含 Generation failed
prediction 是否为空
completion_tokens 是否接近输出上限
score 是否全部为 0
cost 是否为 0
```

如果模型是 reasoning model，尤其要关注：

- 思考内容是否混入最终答案；
- `Answer: X` 格式是否稳定；
- 是否因默认 max tokens 太小而被截断；
- 是否兼容 `temperature`、`top_p`。

### 第七步：全量采集

冒烟通过后，将 demo 配置改成正式配置：

```yaml
run:
  demo_mode: false
  output_dir: results_my_eval
  overwrite: false
```

然后执行：

```bash
python -m data_collector.cli run config/eval_my_model.yaml -y
```

如果修改了 prompt、判分逻辑或数据，必须使用新的 `output_dir` 或显式 `overwrite: true`，否则旧结果会被保留或被错误复用。

### 第八步：构建分析配置

复制现有配置作为起点：

```text
/Users/augustus/Desktop/LLMBenchmark/config/baseline_config_edge.yaml
```

修改：

- 结果目录指向 `results_my_eval/bench`；
- 数据集列表；
- 模型列表；
- 成本口径；
- train/test 比例和 seed。

然后可以调用：

```python
from baselines import BaselineDataLoader, BaselineAggregator

loader = BaselineDataLoader(
    "/Users/augustus/Desktop/LLMBenchmark/config/baseline_my_eval.yaml"
)

records = loader.load_all_records()
agg = BaselineAggregator(records, data_loader=loader)

perf_df, cost_df = agg.to_summary_table(
    cost_metric="total_cost",
    random_seed=42,
)

agg.print_summary_tables(score_as_percent=True)
```

建议至少输出：

- 每个候选模型的单独准确率；
- AVG；
- Random Router；
- Max Expert；
- Category Expert；
- Oracle；
- 总成本；
- 准确率-成本散点图或 Pareto 曲线。

### 第九步：再做路由算法

等答案矩阵完整、判分口径确认之后，再进入 Adaptor 和路由训练。

大致顺序：

```text
BaselineDataLoader
  → 按 prompt 划分 train/test
  → Adaptor 转格式
  → 训练路由器
  → 测试集预测模型选择
  → 回查答案矩阵
  → 计算准确率和成本
```

不要一开始就训练 RouterDC、EmbedLLM 等算法。先确认单模型和静态基线结果合理，否则路由算法结果很难解释。

---

## 九、最容易踩的坑

1. **数据目前是 LFS pointer**
   - 必须先 `git lfs pull`，否则 Evaluator 读不到真实数据。

2. **没有 `results/bench`**
   - 无法直接复现官方论文指标和同事历史报告。

3. **当前环境缺主要依赖**
   - `openai`、`pandas`、`datasets`、`torch`、`transformers`、`loguru` 等尚未安装。

4. **命令必须从仓库根目录执行**
   - 否则模块路径和配置路径可能出错。

5. **Generator 失败可能表现为 0 分**
   - 必须检查 `Generation failed`，不能只看最终 accuracy。

6. **默认不覆盖旧结果**
   - 修改 prompt、判分器或数据后，旧结果可能继续被使用。

7. **demo 结果默认不进正式分析**
   - 正式跑时记得切换 `demo_mode: false`。

8. **同一 dataset/split/model 只取最新文件**
   - 多次运行不会自动合并。

9. **索引是 1-based**
   - 自定义矩阵对齐不要硬编码 0 或 1 的假设。

10. **没有统一 max output token 配置**
    - 推理模型可能截断，需要通过 `extra_body` 或服务端参数验证。

11. **LLM Judge 默认是内网地址**
    - 外部环境需要重新配置 `GRADER_*`。

12. **币种和成本口径不统一**
    - 官方多为 USD，同事报告是 CNY，本地模型成本暂按 0。

13. **Max Expert、Category Expert、Oracle 不是可部署结果**
    - 它们使用测试标签选择模型或是事后知道正确答案。

14. **训练/测试必须按 prompt 划分**
    - 不能直接按模型输出记录随机切分，否则同一道题会同时出现在训练集和测试集。

15. **9B/14B 版本未完全同步**
    - 复现前必须统一工具、配置和报告口径。

---

## 十、推荐的学习顺序

如果想快速掌握同事的二次开发仓库，建议按这个顺序阅读：

1. 先看整体说明：

```text
/Users/augustus/Desktop/LLMBenchmark/docs/pipeline-and-edge-commits.md
```

2. 再看采集执行入口：

```text
/Users/augustus/Desktop/LLMBenchmark/data_collector/cli.py
/Users/augustus/Desktop/LLMBenchmark/data_collector/planner.py
/Users/augustus/Desktop/LLMBenchmark/data_collector/runner.py
/Users/augustus/Desktop/LLMBenchmark/generators/generator.py
```

3. 然后看答案矩阵和基线：

```text
/Users/augustus/Desktop/LLMBenchmark/baselines/schema.py
/Users/augustus/Desktop/LLMBenchmark/baselines/data_loader.py
/Users/augustus/Desktop/LLMBenchmark/baselines/aggregators.py
```

4. 再看同事新增的端侧路由：

```text
/Users/augustus/Desktop/LLMBenchmark/baselines/device_llm_router.py
/Users/augustus/Desktop/LLMBenchmark/tools/eval_device_llm_router_3model.py
/Users/augustus/Desktop/LLMBenchmark/tools/eval_3model_full.py
```

5. 最后再进入具体路由算法：

```text
/Users/augustus/Desktop/LLMBenchmark/baselines/adaptors/
```

对于当前目标，最合理的落地路径是：

```text
先拉 LFS 数据
  → 安装主 Collector 依赖
  → 用 demo 配置接入一个新模型
  → 跑 1 个模型 × 1~2 个数据集 × 6 道题
  → 检查失败率、截断、token 和 cost
  → 再扩展到完整数据集
  → 最后加入多模型答案矩阵和路由分析
```

---

## 附：重要提醒

目前尚未执行依赖安装、完整数据拉取或真实评测。文中所有报告数字都应视为仓库历史结果，而不是在当前机器上重新跑出的新模型结果。
