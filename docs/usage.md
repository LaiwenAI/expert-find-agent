# 使用与复现

Python 3.11；语料处理、BM25、TF-IDF 和代理 benchmark 仅依赖标准库。模型 agent 依赖 `laiwenai==0.0.12`，从当前工作目录 `.env` 读取配置。所有命令在仓库根目录执行。

```bash
PYTHONPATH=src python3 -m expert_search prepare
PYTHONPATH=src python3 -m expert_search index
PYTHONPATH=src python3 -m expert_search search '机器人运动控制' --method bm25-max -k 5
PYTHONPATH=src python3 -m expert_search search '机器人运动控制' --method bm25-vote -k 5
PYTHONPATH=src python3 -m expert_search search '机器人运动控制' --method tfidf-max -k 5
PYTHONPATH=src python3 -m expert_search agent '寻找机器人运动控制和强化学习方面的专家' -k 3 --candidates 8
PYTHONPATH=src python3 -m expert_search benchmark
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 方法与适用范围

- BM25-max：先按证据检索，每名专家取最强证据分数。
- BM25-vote：每名专家累加最强五条证据，属于截断投票式实现；没有声称严格复现 Balog 的生成模型。
- TF-IDF-max：使用 BM25 的平滑 IDF、对数 TF、证据长度平方根归一化，取专家最强证据。该方法是 TF-IDF 变体，不是完整向量余弦实现。
- hybrid-evidence-tree：模型生成双语技术查询；用 reciprocal rank fusion 合并召回；展示“专家→字段/论文→证据叶子”的树，由模型选择叶子并阅读全文片段，最后按证据生成推荐。迁移 PageIndex 的树导航、查证思想；首阶段仍有 BM25，因此是混合方法。当前每分支最多展示前 40 叶、总共最多读取 30 叶，每叶最长 6,000 字符；这些截断可能影响召回。

所有引用均检查证据存在和专家归属；这并不保证模型关于相关性的解释正确。trace 记录模型各阶段输出与耗时，usage 记录模型返回的 token 使用。默认命令会向 `.env` 配置的模型服务发送需求和候选人的科研资料。

## 目前验证

`artifacts/agent-smoke.json` 保存一次真实模型运行，返回三名专家及证据。此单例证明流程可运行，不能证明推荐质量。代理 benchmark 排名和配置分别保存在 `artifacts/benchmark/runs.jsonl` 和 `manifest.json`。

## 后续研究方向

当前统一比较已纳入 TF-IDF 和树 agent。后续可扩展稠密检索、真实需求人工评测、模型输出结构恢复和树分支截断消融。论文标题作者恢复仅作代理任务，不等同于产业专家需求匹配。

## 实验与审计命令

```bash
PYTHONPATH=src python3 -m expert_search agent-benchmark
PYTHONPATH=src python3 -m expert_search demand-cases
PYTHONPATH=src python3 -m expert_search report
PYTHONPATH=src python3 -m expert_search verify
```

agent-benchmark 默认评测基线前 10 条查询；保存每条结果并可恢复，输入 manifest 指纹不匹配会报错。report 在同一已完成查询子集比较方法，运行中生成的表格属于阶段结果。verify 重算指标，检查专家 ID、排名去重、证据来源和留出规则。demand-cases 默认运行已抽样查询的前三条，保存各基线和 agent 的原始结果，只用于定性检查。

## 快速研究方向接口

```bash
PYTHONPATH=src python3 -m expert_search fast-agent '机器人控制与强化学习' -k 5
PYTHONPATH=src python3 -m expert_search fast-agent '寻找研究「机器人」和「人工智能」的专家' -k 5 --explain
```

输出 NDJSON 事件：先返回检索结果，启用 explain 后再输出模型解释。首批时间、索引初始化时间和解释总时间分别报告；不能把首批延迟当完整解释延迟。该入口只搜索中文研究方向字段，显式「词条」按公开字段精确联合匹配；普通文本按字段 BM25 排序。它不等同于已核实专家能力，也未覆盖全部论文资料。

可加 `--full-evidence`：普通文本检索先输出领域字段结果，再查询全部已索引画像与论文证据，输出 `full_evidence` 事件。排名可能更新；解释使用最终阶段结果，不引用已经被替换的候选。显式「词条」条件保留精确研究方向语义，不自动用论文相似度覆盖条件。

```bash
PYTHONPATH=src python3 -m expert_search fast-agent '机器人运动控制与强化学习' --full-evidence --explain
```

全语料阶段是词法证据排序，尚不是已验证的语义推理。首批结果与完整解释的延迟必须分开报告。

`--reasoning-effort none` 可显式关闭模型额外推理，仅用于启用解释时的模型请求；不改变检索规则，不写入证据输入。当前保持默认未指定档位，待完整对照确认质量后再决定默认配置。
## 逻辑检索与范围控制诊断

ComLQ 官方归档按固定提交与 SHA-256 下载；保留逻辑类型，不将答案字段传入检索查询：

```bash
PYTHONPATH=src python3 -m expert_search.research_cli comlq-lexical
```

该命令运行 BM25 与 TF-IDF 文档检索基线并按逻辑类型汇总，不表示产业专家匹配质量。原仓库未发现明确数据许可，归档不进入我们的源码发布包。

n-Clue 的官方冻结榜单可独立复算：

```bash
python3 experiments/reproduce_nclue.py
```

此命令校验固定上游版本；只复算已发布榜单，不运行我们的检索方法，不测量推理延迟。

范围控制合成试验保留原需求未覆盖项，反事实由代码只替换一个主题：

```bash
PYTHONPATH=src python3 -c "from expert_search.synthesis_v2 import build; print(build(n_demands=20, output='artifacts/synthesis-scope-v5', protocol='scope-v5'))"
PYTHONPATH=src python3 -m expert_search.scoped_export
PYTHONPATH=src python3 -m expert_search.scoped_binding
```

后两项分别导出不含原文的本地 silver 审阅候选，以及单独审核联合应用关系。样本不是完整产业需求金标；分歧保留为未决，不赋予确定的关系得分。

## 直接论文—专家匹配基准

```bash
PYTHONPATH=src python3 -m expert_search.cof_data
PYTHONPATH=src python3 -m expert_search.research_cli cof-sigir-lexical
PYTHONPATH=src python3 -m expert_search.research_cli cof-sigir-dense
```

第一项下载并核验 CoF 官方数据归档、审查候选标注及精确文本复制。后两项在 SIGIR 的完整已判候选上运行词法和冻结 API 向量对照；标签不进入编码输入。向量结果使用预计算查询，评分耗时不等于线上编码加检索。数据适配和许可范围见 `docs/exploration/cof-datasets.md`。


配对证据预算分析（只读取已完成记录，不调用模型）：

```bash
PYTHONPATH=src python3 -m expert_search.packing_analysis --dataset SIGIR --serialization inline
PYTHONPATH=src python3 -m expert_search.packing_analysis --dataset SIGIR --serialization shared
PYTHONPATH=src python3 -m expert_search.packing_analysis --dataset NIPS --serialization shared
```

分析器重新计算来源标签上的指标，检查候选池、匿名编号、排列及失败回退。缺少任何计划记录时仅输出完整性状态，不用已完成子集替代全量比较；完整后报告配对查询 bootstrap、均值/中位数/P95、失败与限流等待。区间跨零不证明等价，模型调用耗时不等于端到端延迟。


新增的经典基线与索引实验（无神经模型训练）：

```bash
PYTHONPATH=src python3 -m expert_search.cof_language_benchmark --dataset SIGIR
PYTHONPATH=src python3 -m expert_search.cof_language_benchmark --dataset NIPS
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/compiled_lm_benchmark.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/teacher_compiled_lm.py
PYTHONPATH=src python3 experiments/judgment_budget.py
```

`CompiledCandidateLM.save(path, source_sha256)` 保存无需 pickle 的倒排数值索引；`CompiledCandidateLM.load(path, expected_source_sha256)` 检查快照、数组哈希、范围与排序。预编译只改变经典候选语言模型的执行方式，不改变评分公式；真实教师计时不提供无标签条件下的质量结论。

本地冻结编码及合成候选准备使用 local 可选环境：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-local/bin/python -m expert_search.cof_late_interaction --dataset SIGIR
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-local/bin/python -m expert_search.cof_late_interaction --dataset NIPS
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-local/bin/python -m expert_search.semantic_synthesis_pool
PYTHONPATH=src python3 -m expert_search.semantic_synthesis_pool --execute
```

最后一条执行真实模型调用。候选准备本身只进行本地检索，固定二十条原始需求，用题名检索研究领域，并为同一专家补充词法前三证据及中英研究领域；随后仍需 scope-v5 的完整查询和反事实审核。相似度不作为标签。与旧版同时改变来源策略，不能把接受率变化单独归因于语义检索。运行这类生成任务时，应与限流敏感的网络计时实验分开。


逐条件复审与审阅数据导出：

```bash
PYTHONPATH=src python3 -m expert_search.scoped_dataset_audit
PYTHONPATH=src python3 -m expert_search.scoped_dataset_audit --export
PYTHONPATH=src python3 -m expert_search.compiled_expert_search '机器人运动控制与路径规划' -k 5
```

复审只处理两批已有 silver 候选，逐条件 ID 和唯一引用检查不等于人工语义金标。导出重新验证源文件与原始模型输出，输出不包含教师/需求原文和模型轨迹。完整合成审阅产物仍为本地候选，未授权为第三方语料再分发。


审阅数据的分组与来源支持诊断：

```bash
PYTHONPATH=src python3 -m expert_search.scoped_dataset_audit --export
PYTHONPATH=src python3 -m expert_search.grouped_dataset
PYTHONPATH=src python3 -m expert_search.scoped_silver_benchmark --split test
```

重新导出后应重新生成切分；加载器核验 split 与 pairs 的哈希，以及每条完整输入来源的前缀内容、哈希和专家归属。该诊断不是专家相关性或实际能力评价，详见 `docs/scoped-dataset-card.md`。


### 冻结交叉编码器及语义传输探索

先按现有 CoF 流程准备 `packing-shared-v2` 输入。交叉编码器首次运行下载模型，需安装可选依赖；模型不进入源码包。

```sh
python -m pip install -e '.[rerank]'
PYTHONPATH=src python -m expert_search.cof_crossencoder --dataset SIGIR --model ms-marco-TinyBERT-L-2-v2 --mode source-max
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/semantic_transport.py --dataset SIGIR
```

语义传输另需 `.[local]`。固定协议位于 `docs/exploration/semantic-transport-protocol.json`；完整工作目录保留该协议。两类实验分别写入 CoF 工件目录，核与向量构建成本不含于在线评分计时。来源反事实修正复审命令为 `python -m expert_search.source_counterfactual --reaudit`，依赖原始本地 v1 工件，不构成独立可下载数据集。


### 精确有序前 K 的执行优化

持久化教师查询入口已使用向量化上下界与精确重放，保持经典模型的前 K 顺序及返回分数，包含 ID 平分规则。它仍访问所有活跃倒排项，并不验证专家能力。完整教师库配对复验：

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/certified_topk_benchmark.py
PYTHONPATH=src python3 experiments/semantic_transport_analysis.py
```

前者依赖已有教师编译索引及20条原始需求，写入 `artifacts/certified-topk-v2/`。后者重验 CoF 两个完整查询集的来源指标并计算探索性配对区间，不能用平均P@5相同代替排序相同。

需求关联来源控制提供显式本地准备路径：

```sh
PYTHONPATH=src python experiments/demand_aligned_source_controls.py --local
```

需安装 `.[local]`，首次下载多语言 MiniLM ONNX 导出，最多128 token。该步骤只准备来源，不产生有效审核标签；`--execute` 才运行双模型构造审核，仍需网关支持协议里的 GPT-4o 和 GPT-6-Luna。默认 API embedding 路径与 `--local` 分属不同工件目录，不会静默替换后端。

若审核后端不可用，可单独保留生成可行性结果：`PYTHONPATH=src python3 experiments/demand_aligned_source_controls.py --local --generate-only`。该命令使用已准备的本地来源，不需要运行环境安装FastEmbed，但需要laiwenai及可用GPT-4o；输出只写generation-only子目录，既不替代双模型审核，也不作为可发布标签。


Apple Silicon 的冻结Qwen基线需安装 `.[mlx]`。依次运行 `experiments/prepare_qwen3_reranker.py` 和 `experiments/qwen_reranker_benchmark.py --dataset SIGIR`。当前入口使用4GiB内存上限及128MiB缓存，结果写入qwen-reranker-mlx-v2。`experiments/qwen_reranker_analysis.py` 重验来源、候选映射及逐查询指标，未完成的运行只输出完成状态。


来源优先的构造实验入口为 `experiments/witness_first_controls.py`，依赖已准备的local-v4来源与多语言向量。它保留联合关系规划、独立来源检索及精确见证选择的各阶段响应。`experiments/witness_control_auxiliary_audit.py` 对两种构造流程的全部结构成功样本进行GPT-4o-mini辅助审核。当前四例结果为三例分歧、一例未决，后续数据导出仍需核查支持状态与真实需求关联。

Decisions适配器在新版SDK中调用 `client.decisions.create`，旧版SDK通过同一原生端点发送请求。当前网关实测返回400，错误会保留给调用方。


前缀缓存配对试验运行 `experiments/qwen_prefix_pilot.py`，使用固定的每数据集前六条协议查询、两次重复。加入 `--fp32` 可独立测量FP32模型。分析入口 `experiments/qwen_prefix_analysis.py` 使用相同选项，要求完整24次记录。FP32与BF16的原始记录分别保存在独立目录。


独立复审原文包由 `PYTHONPATH=src python3 experiments/build_blind_review.py` 生成。当前输出8份待填审核表，覆盖4个构造案例的两种来源快照。审阅输入使用重新编号的来源，隐藏模型判定、生成器理由和full/control条件名称。原始来源映射单独保存在private-mapping.json，审核者提交前应只读取inputs.jsonl与说明。原文包保存在artifacts目录，当前人工标签数为0。


填写后的完整审核文件可用 `PYTHONPATH=src python3 experiments/validate_blind_review.py <submission.jsonl>` 校验。校验输入快照哈希、项目完整性、来源与条件文本、审核者标识、三项条件标签，以及支持性引文和需求锚点的唯一位置。输出保留提交文件哈希，便于后续复核。空白模板会被拒绝。


较大CoF集合的固定语言模型检查使用 `experiments/cof_scale_check.py --dataset KDD`，也支持SciRepEval。`cof_scale_analysis.py` 按原始协议与排除精确重复查询的集合分别复算。完成词法基线后运行 `cof_scale_comparison.py`，对BM25-max、BM25-top3逐查询比较P@5与nDCG@10，并输出探索性配对区间。缺失任何计划查询时只输出完成状态。DOI查询ID采用URL编码文件名。

Decisions API 恢复后，可使用已配置的 `.env` 和 `gpt-6-luna` 运行真实来源视图试验：

```sh
PYTHONPATH=src .venv-local/bin/python experiments/decisions_blind_pilot.py
PYTHONPATH=src .venv-local/bin/python experiments/decisions_batch_comparison.py
```

两者依赖本地 `artifacts/blind-source-review-v1/inputs.jsonl`。第二个试验对同一视图随机安排逐项与合批的先后顺序，保存逐请求响应、延迟与标签一致性；模型输出不构成人工金标准。

固定排名融合探索（需要 KDD、SciRepEval 的 `scale-check-v1` 结果及本地 BM25 索引）：

```sh
PYTHONPATH=src python3 experiments/cof_rank_fusion.py
```

方案固定为 Model1 与 BM25-top3 等权 reciprocal rank fusion，常数 60；保存完整候选排名、两个原方法与融合方法的逐查询指标及配对 bootstrap。该方案是在查看基线均值后提出的探索，不能标注为未见测试集上的确认实验；当前复用缓存 Model1 排名，不报告在线延迟。

词级最强证据聚合探索：

```sh
PYTHONPATH=src python3 experiments/cof_max_term.py --dataset NIPS
PYTHONPATH=src python3 experiments/cof_max_term.py --dataset KDD
PYTHONPATH=src python3 experiments/cof_max_term.py --dataset SciRepEval
```

该方案对每位专家的每个词预计算其论文中的最大 BM25 贡献，查询时求和；结果目录为对应数据集的 `max-term-v1`。它检验离线编译的词级聚合，不可把跨论文的词匹配解释为联合能力证据。计时排除索引初始化。

需求先行的数据构造准备：

```sh
PYTHONPATH=src .venv-local/bin/python experiments/demand_first_facets.py
PYTHONPATH=src .venv-local/bin/python experiments/prepare_demand_first_pool.py
```

先等待第一步完整结束再运行第二步。抽取结果在 `artifacts/demand-first-facets-v2`；研究主题驱动的教师证据候选在 `artifacts/demand-first-pool-v1`。这些候选未经过独立相关性和能力标注，不能直接用作 benchmark 金标准。

固定需求与教师候选，改变论文证据选择的匹配实验：

```sh
PYTHONPATH=src .venv-local/bin/python experiments/demand_first_semantic_sources.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_demand_first_pool.py --semantic
```

需先完成原始 `audit_demand_first_pool.py` 以冻结 13 个试验任务。多语言模型仅用于同一教师论文排序，使用已有本地缓存，无模型训练；保存模型文件、编码输入与向量摘要。审查输出仍只是单模型判断，不能作为最终数据标签。

从完整真实需求中筛选可构造候选，并生成带论文引文的窄能力条件：

```sh
PYTHONPATH=src python3 experiments/mine_demand_topic_overlap.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_topic_overlap.py
PYTHONPATH=src python3 experiments/prepare_paper_backed_overlap.py
PYTHONPATH=src .venv-local/bin/python experiments/generate_paper_backed_tasks.py
```

每步完成后再执行下一步。所有阶段均保留拒绝及未解决结果。最后一步只是生成待审核条件，不能直接作为数据集标签；本地结果包含私有原文，不随源码审阅包分发。

对全部构造成功的论文能力条件执行双模型盲审：

```sh
PYTHONPATH=src .venv-local/bin/python experiments/audit_paper_backed_tasks.py
```

审查输入包含原需求、完整能力条件及冻结论文窗口，隐藏生成器理由、引文和预期标签。GPT-4o 与 GPT-6-Luna 必须分别确认需求关联与证据支持，且引文通过唯一原文验证，才能标为待进一步处理的双模型银标候选。两者同属模型家族，仍不构成人工金标准。

冻结 Qwen3 Embedding 0.6B 基线（Apple Silicon MLX 环境）：

```sh
python3 experiments/prepare_qwen3_embedding.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-mlx/bin/python experiments/cof_qwen_embedding.py
```

下载固定官方 revision，校验权重 LFS 哈希；不修改权重。编码预算固定 512 token，批量 8，使用最后真实 token 的 FP32 归一化向量。编码块保存在对应 CoF 数据集 `qwen-embedding-v1`，输入摘要变化会拒绝复用；缓存查询的排序耗时不包含在线查询编码。编码期间 GPU 内存上限 4 GiB、缓存上限 128 MiB。

需求无关的专家语义证据目录原型（固定 21 位教师可行性队列）：

```sh
PYTHONPATH=src .venv-local/bin/python experiments/build_expert_semantic_tree.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_expert_tree_nodes.py
```

等待第一步完整结束再运行第二步。每位专家最多六个“问题—方法—对象”节点，只能引用本人的真实来源片段。构造结果的结构和跨度验证不等于摘要正确；第二步单独判断节点描述是否获得所引片段支持。未获支持的摘要不应作为已验证能力，原始来源仍保留回退。

全库粗到细路由对照：

```sh
PYTHONPATH=src python3 experiments/prepare_global_expert_roots.py
PYTHONPATH=src python3 experiments/prepare_global_expert_branches.py
```

根节点优先保留研究领域及前三篇论文标题，缺失时保留原始资料回退；分支对照保留每条非空原始来源的1200字符前缀。编码脚本为 `global_expert_root_benchmark.py` 与 `global_expert_branch_benchmark.py`，依赖冻结Qwen3 Embedding及CUDA环境。Slurm提交脚本为 `cluster_global_roots.sbatch`、`cluster_global_branches.sbatch`，须先将输入与代码传到计算节点大盘，并从登录节点提交。源文本仅作路由提示；前缀和token截断都不能保证完整证据覆盖。


同节点功能表示诊断（本地冻结模型需预先准备）：

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mlx/bin/python experiments/compare_functional_routes.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mlx/bin/python experiments/compare_functional_routes.py --condition-diagnostic
```

第二条使用已知银标条件，仅用于区分目录表示与完整需求映射问题，不作为部署质量。两种模式各有独立输入哈希、向量与结果目录，拒绝变动输入复用旧缓存。


目录候选的原文适用性核验与单独模型复核（需要已有类型化路由、需求分项与原始本地证据；读取 `.env`，调用gpt-6-luna及gpt-4o）：

```sh
PYTHONPATH=src .venv-local/bin/python experiments/verify_typed_route_sources.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_source_applicability.py
```

核验保留原始响应，拒绝未知归属及非原文引文；结构有效仅证明引用定位，不证明方法适用。复核结果单独保存，未解决结果不作为负例，均不充当评测金标。


固定片段ID定位及组件/整项分离复核：

```sh
PYTHONPATH=src .venv-local/bin/python experiments/verify_typed_route_sources.py --span-ids
PYTHONPATH=src .venv-local/bin/python experiments/audit_source_applicability.py --input artifacts/span-applicability-route-v1 --output artifacts/span-applicability-review-v1
PYTHONPATH=src .venv-local/bin/python experiments/audit_source_applicability.py --component-review --input artifacts/span-applicability-route-v1 --output artifacts/span-component-review-v2
```

组件复核是查看前期复核分歧后制定的探索性口径，不能选择较有利结果当作金标。具体组件联系和整项需求覆盖分别报告，重复组件尚未去重，不作完整需求覆盖率。


需求操作拆分与成对模型审查（仅输入固定九条企业需求，不输入候选专家）：

```sh
PYTHONPATH=src .venv-local/bin/python experiments/plan_demand_operations.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_demand_operations.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_demand_operations.py --model gpt-6-luna --output artifacts/demand-operation-audit-luna-v1
PYTHONPATH=src .venv-local/bin/python experiments/compare_operation_reviews.py
```

技术操作与集成、部署、验证义务分开保存；结构校验拒绝重复同范围操作、未知依赖和变动原文。逐字锚点存在不等于语义完整。模型审查的一致性不是正确率，不自动删除争议操作或据“遗漏”报告补写要求；原始响应全部保留。


公开CoF来源向量分组上界的成本对照（需先物化官方Qwen CUDA向量缓存）：

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_geometric_bounds.py --dataset SIGIR
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_geometric_bounds.py --dataset NIPS
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_geometric_bounds.py --dataset SIGIR --stable-dots
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_geometric_bounds.py --dataset NIPS --stable-dots
```

稳定点积模式用相同逐行求和规则核对顺序，单独保存结果；不把跳过成员数量当成速度收益，不将经验一致性解释为形式认证。此分组方法在当前两套数据上更慢，保留作对照。


成熟Faiss HNSW与精确来源预算对照（官方冻结Qwen CUDA缓存需已准备）：

```sh
.venv-local/bin/python -m pip install -e '.[ann]'
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_hnsw.py --dataset SIGIR
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_hnsw.py --dataset NIPS
```

固定来源预算32/128/512；恢复专家后重算其全部论文。完整相似度前五保留率、来源前十包含率与公开标签精度分开保存；不向排名追加未召回专家。Faiss官方实现参考：https://github.com/facebookresearch/faiss/wiki/Faiss-indexes 。该路径是成熟对照，非原创方法；计时排除在线编码与证据核验。


候选资格ID感知的来源扩展、直接扫描及选择性切换对照：

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_eligible_routing.py --dataset SIGIR
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_eligible_routing.py --dataset NIPS
PYTHONPATH=src .venv-local/bin/python -m unittest discover -s tests -p test_source_routing.py
```

公开候选ID作为资格过滤条件，相关性数值仅用于最终评测；不能在部署中用未知相关性标注推导资格池。共享资格准备、各路径局部耗时分别报告，不称端到端加速。扩展按合格专家数量停止，无排名最优性保证；回退不隐去。


真实来源的统一路径贯通测试（21位专家、296条来源，两条复用需求；实际调用gpt-6-luna，需要.env和已验证官方Qwen检查点）：

```sh
.venv-mlx/bin/python -m pip install --index-url https://pypi.org/simple laiwenai==0.0.12 faiss-cpu==1.16.0
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mlx/bin/python experiments/grounded_pipeline_smoke.py
```

该脚本绑定目录队列、输入及代码哈希，保存源码向量和逐阶段真实运行记录。已完成记录被缓存，不把重新读取当作新的在线时延。OwnedVectorIndex和GroundedPipeline可独立导入；当前规划及适用性仍为模型原型，最终结果标为provisional或partial，引用检查不是语义正确性保证。模型初始化、来源编码、在线规划、编码、检索、读取和核验分别报告。


紧凑单候选/三候选核验对照（固定真实贯通测试的分项与来源，实际调用gpt-6-luna）：

```sh
PYTHONPATH=src .venv-local/bin/python experiments/compare_compact_assessment.py
PYTHONPATH=src .venv-local/bin/python experiments/compact_assessment_stability.py
PYTHONPATH=src .venv-local/bin/python -m unittest discover -s tests -p test_batch_applicability.py
```

每种方式两轮、交替运行顺序。批量组件使用 `BatchApplicability.assess_many`，未替换GroundedPipeline默认逐候选核验。跨模式与同模式重复均存在标签不稳定，不能把结构通过或请求减少当作质量等价。

需求先行的全库论文构造流程（GPU检索在计算节点执行，生成/审查使用本地gpt-6-luna及gpt-4o）：

```sh
# 本地先从既有固定20条需求生成冻结输入：
PYTHONPATH=src .venv-local/bin/python experiments/prepare_demand_first_queries.py
# 将输入同步计算节点；准备既有全库来源向量后，在Slurm计算节点执行：
PYTHONPATH=src venv/bin/python experiments/retrieve_demand_first_papers.py
# 结果回传本地后：
PYTHONPATH=src .venv-local/bin/python experiments/prepare_demand_first_jobs.py
PYTHONPATH=src .venv-local/bin/python experiments/generate_demand_first_tasks.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_paper_backed_spans.py --input artifacts/demand-first-paper-jobs-v1/jobs.jsonl --generation artifacts/demand-first-paper-generation-v1 --output artifacts/demand-first-paper-span-audit-v1
PYTHONPATH=src .venv-local/bin/python experiments/paper_backed_controls.py --input artifacts/demand-first-paper-jobs-v1/jobs.jsonl --audit artifacts/demand-first-paper-span-audit-v1 --generation artifacts/demand-first-paper-generation-v1 --output artifacts/demand-first-paper-controls-v1 --replacement-corpus artifacts/corpus
PYTHONPATH=src .venv-local/bin/python experiments/audit_paper_controls.py --input artifacts/demand-first-paper-controls-v1 --output artifacts/demand-first-control-blind-audit-v1
PYTHONPATH=src .venv-local/bin/python experiments/export_paper_controls.py --audit artifacts/demand-first-control-blind-audit-v1 --controls artifacts/demand-first-paper-controls-v1 --output artifacts/dataset-release/demand-first-paper-controls-silver-v1
PYTHONPATH=src .venv-local/bin/python experiments/combine_paper_controls.py --include-demand-first
```

20→9→4→3→1的全部构造漏斗保留。新版合并5对10视图，旧四对数据保持原版本。详见[控制数据卡](paper-controls-data-card.md)；双模型一致不替代人工金标，不把检索命中或不可构造直接转成能力标签。


冻结Qwen语义与BM25词法融合的质量对照：

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_source_fusion.py
PYTHONPATH=src .venv-local/bin/python experiments/diagnose_source_fusion.py
PYTHONPATH=src .venv-local/bin/python experiments/evaluate_fusion_policy.py --datasets SIGIR NIPS
# 扩展所需BM25分数先在本地物化，再同步计算节点：
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/prepare_lexical_snapshots.py
# 计算节点编码及融合入口见cluster_fusion_kdd.sbatch、cluster_fusion_scirepeval.sbatch
```

全部固定变体保留。资格池≤32时来源top3、否则Qwen的规则在扩展融合输出前冻结；不是依据扩展结果选参。所有时间仍须区分编码、BM25及共享缓存，来源贡献不等于已证实能力。NIPS与SIGIR结果方向不同，不据单数据集优势作总体结论。

细粒度证据阶段的辅助开发基准（非企业专家适用性评价）：

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mlx/bin/python experiments/evidencebench_pilot.py
# 以下会实际调用已配置的gpt-6-luna API：
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mlx/bin/python experiments/evidencebench_decisions.py
PYTHONPATH=src .venv-local/bin/python experiments/evidencebench_set_selector.py
```

原文按固定上游版本下载，不进入源码包。已有结果按协议读取，不把缓存重读当作新在线时延。见[辅助数据卡](evidencebench-card.md)。新增顶会PDF放在用户指定的literatures/method目录，来源与哈希见[全文审查记录](exploration/topconf-review-20261117.md)；PDF不复制进软件仓库。

辅助来源角色编译开发实验（需先完成EvidenceBench pilot生成来源向量；会调用gpt-6-luna，原文离线传入模型）：

```bash
PYTHONPATH=src .venv-mlx/bin/python experiments/evidencebench_role_index.py
```

输出协议锁定源码和来源哈希；编号分区不完整即使用原始语义回退。该版本没有证明质量收益，不是默认搜索流程。

固定原始测试293条的来源角色实验：

```bash
PYTHONPATH=src .venv-local/bin/python experiments/evidencebench_role_test.py --stage compile
# 以下两步需CUDA环境，集群分别见cluster_evidencebench_test.sbatch与cluster_evidencebench_evaluate.sbatch
PYTHONPATH=src python experiments/evidencebench_role_test.py --stage vectors
PYTHONPATH=src python experiments/evidencebench_role_test.py --stage evaluate
```

编译只输入来源句子；评价才使用假设和方面标注。协议锁定来源与实现，已有产物可续用。第三方原文和模型响应不随源码包分发。

真实来源固定3片段预算诊断（复用本地五对银标，含API调用；不是专家排名评测）：

```bash
PYTHONPATH=src .venv-mlx/bin/python experiments/enterprise_role_budget.py
PYTHONPATH=src .venv-mlx/bin/python experiments/enterprise_query_budget.py
```

前者离线编译单条来源角色，后者从原需求进行来源盲规划。协议在选择与指标前固定；标签只用于选择后引用保留诊断。原文与返回轨迹保留本地，不进入源码包。

相同需求的匿名片段视图评审、全部分歧复核及一次相同输入重复（均调用gpt-6-luna；模型评分不是金标）：

```bash
PYTHONPATH=src .venv-local/bin/python experiments/enterprise_span_assessment.py
PYTHONPATH=src .venv-local/bin/python experiments/enterprise_span_disagreements.py
PYTHONPATH=src .venv-local/bin/python experiments/enterprise_disagreement_audit.py
PYTHONPATH=src .venv-local/bin/python experiments/enterprise_span_repeat.py
PYTHONPATH=src .venv-local/bin/python experiments/analyze_enterprise_span_stability.py
```

分歧输入为`artifacts/enterprise-span-assessment-v1/disagreements.json`，保存所有压缩/完整上下文可操作判断翻转；不是挑选正确样本。重复实验严格检查父协议和相同输入哈希。源码包含视图核验模块与归属/偏移测试，原始来源和模型响应保持本地。

条件性方法迁移案例构造（完整保留20条固定需求，所有模型调用均为gpt-6-luna）：

```bash
PYTHONPATH=src .venv-local/bin/python experiments/generate_transfer_cases.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_transfer_cases.py
PYTHONPATH=src .venv-local/bin/python experiments/transfer_source_controls.py
PYTHONPATH=src .venv-local/bin/python experiments/export_transfer_cases.py
```

输入为已有需求先行检索任务；来源控制需要完整本地语料catalog。输出本地`transfer-candidates-v1`，标签范围和未完成审核见[迁移案例数据卡](transfer-cases-card.md)。不把未知事实来源视作负能力标签，不把条件性迁移解释视作已实现未来项目。

全库归属来源检索基线（需要既有冻结来源矩阵；编码需CUDA，索引/比较仅用CPU）：

```bash
PYTHONPATH=src python experiments/global_owned_routing.py --stage encode
PYTHONPATH=src python experiments/global_owned_routing.py --stage benchmark
PYTHONPATH=src python experiments/global_owned_effort.py
PYTHONPATH=src python experiments/verify_global_owned_routing.py
PYTHONPATH=src python experiments/verify_global_owned_routing.py --directory artifacts/global-owned-effort-v1
```

对应Slurm脚本为`cluster_global_owned_encode.sbatch`（ws5 GPU）、`cluster_global_owned_routing.sbatch`和`cluster_global_owned_effort.sbatch`（ws6 CPU）。`ExactOwnedVectorIndex`为最大来源点积的高效全量参考；默认ANN与固定efSearch512对照均为成熟近似搜索基线。比较仅衡量原始语义排序保留及局部耗时，不能当作真实专家精度或最终代理时延。

## 新原始需求评测集

范围、来源与未判定标签规则见[数据卡](fresh-evaluation-data-card.md)。已有本地原始数据及冻结生成工件时：

```bash
PYTHONPATH=src python experiments/export_fresh_evaluation.py
PYTHONPATH=src python experiments/verify_fresh_evaluation.py
PYTHONPATH=src python experiments/benchmark_fresh_evaluation.py
python experiments/package_fresh_evaluation.py
```

导出依赖先前冻结的选择、候选池和开发标注工件；不是仅从原始数据一步重跑模型的快捷命令。生成入口依次为`prepare_fresh_method_data.py`、`rebuild_fresh_sources.py`、`fresh_method_retrieval.py`（编码）、`fresh_evaluation_pool.py`及`annotate_fresh_evaluation.py`，具体参数以各脚本为准。模型生成会产生调用费用，不应为复核已有工件而重跑。封存集保持未标注。数据压缩包仅本地保存，软件许可不覆盖原始文本。

开发集候选池偏差审查及未标注扩充（不会修改冻结数据）：

```bash
PYTHONPATH=src python experiments/fresh_lexical_pool_audit.py
python experiments/export_fresh_lexical_review.py
python experiments/audit_fresh_annotation.py
```

新增材料位于`artifacts/fresh-lexical-pool-audit-v1/review-additions`，742个视图均未标注，不得继承其他快照的标签或作为已知负例。

短选项接口配对实验入口为`experiments/choice_interface_comparison.py`，固定前20条开发需求、两接口各一次，含付费API调用。第二轮使用同一模块将`ROOT`设为`artifacts/choice-interface-comparison-repeat-v1`，其余参数及输入不变。`experiments/analyze_choice_interface.py`读取两轮冻结输出，计算接口间及接口内重复一致性和按需求分组的耗时bootstrap；不提供语义正确率。全部使用gpt-6-luna，无重试，失败保留。

门控开发审查与实际两阶段对照：

```bash
PYTHONPATH=src python experiments/audit_choice_gating.py
PYTHONPATH=src:experiments python experiments/staged_partial_match_comparison.py
PYTHONPATH=src python experiments/analyze_staged_partial.py
```

第二项使用付费gpt-6-luna API；已有冻结输出时只读取缓存，协议不一致会拒绝覆盖。第三项包含实际门控耗时、失败、已知银标遗漏和2/5/10/20秒截止结果，不将未知标签视为负例。开发实验不支持两阶段加速，该流程不是默认搜索路径。

单次请求只解释最终前三名的实验：

```bash
PYTHONPATH=src python experiments/focused_partial_comparison.py
PYTHONPATH=src python experiments/analyze_focused_partial.py
PYTHONPATH=src:experiments python experiments/audit_focused_witnesses.py
```

第一、三项调用付费gpt-6-luna API。比较固定开发集位置20:40，双臂均读七名候选，只改变解释输出预算，均评价前三名。实现为`FocusedPartialMethodMatcher`，尾部候选保持未评估，不输出能力否定。后置见证审查为同模型诊断，不计入两臂计时，也不是独立金标；当前尚未接入默认搜索。

具体迁移表达消融（开发位置40:60）：

```bash
PYTHONPATH=src python experiments/concrete_partial_comparison.py
PYTHONPATH=src:experiments python experiments/audit_concrete_witnesses.py
PYTHONPATH=src python experiments/analyze_concrete_partial.py
```

前两项为付费API调用；后者使用固定见证规则，不是独立金标，也不计入在线两臂耗时。重复审查使用同一模块，先将`audit_concrete_witnesses.ROOT`设为`Path('artifacts/concrete-witness-repeat-v1')`再调用`run()`，保持冻结生成结果和盲序不变；随后运行`experiments/analyze_concrete_witness_repeat.py`。首轮失败不由重审回填。`ConcreteFocusedPartialMatcher`只增加具体迁移及完整中文表达指令，保留原归属/schema/一次调用/前三名预算；未接入默认搜索，语言和桥接因素尚未分别消融。
