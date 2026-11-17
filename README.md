# 专家搜索 Agent

源码已在[LaiwenAI/expert-find-agent](https://github.com/LaiwenAI/expert-find-agent)公开（MIT），默认分支为`research/prototype-20261117`。当前为研究原型；真实原文数据包仍仅在本地，尚未证明专家精度与端到端效率的联合优势。

面向企业技术需求的 evidence-based expert search 研究原型：从专家论文和简介中定位证据，探索直接能力与方法迁移的判断及排序。沿用 PageIndex 可追溯层次目录的思路，检验全库粗筛、专家能力分支和原文细查的组合。当前 agent 采用 BM25 召回与证据树导航；语义能力目录、适用性协议和全库多分支路由仍在独立实验中验证。

```bash
pip install -r requirements.txt
PYTHONPATH=src python3 -m expert_search prepare
PYTHONPATH=src python3 -m expert_search index
PYTHONPATH=src python3 -m expert_search search '机器人运动控制' -k 5
PYTHONPATH=src python3 -m expert_search agent '寻找机器人运动控制与强化学习专家' -k 3 --candidates 8
```

模型配置见 `.env.example`；请在仓库根目录运行，agent 从本目录 `.env` 读取配置。

- [使用与复现](docs/usage.md)
- [PageIndex 专家索引设计审查](docs/exploration/pageindex-expert-design.md)
- [GPU 集群运行方式](docs/cluster.md)
- [问题定义和论文方法](docs/research-methods.md)
- [Benchmark 结果](docs/benchmark-results.md)
- [真实需求构造与控制实验](docs/paper/draft-v3.md)
- [人工标注协议](docs/annotation-protocol.md)
- [从原始数据重建的新评测集](docs/fresh-evaluation-data-card.md)

Benchmark 比较 BM25-max、BM25-vote、TF-IDF 变体与混合证据树 agent。论文作者恢复属于代理任务，不能作为企业需求匹配准确率。当前版本为研究原型，主要限制是词法候选召回、叶子展示截断及缺少真实需求人工 gold。

## 无训练技术探索与论文

- [研究初稿 v3](docs/paper/draft-v3.md)
- [新增顶会全文审查](docs/exploration/topconf-review-20261117.md)
- [完整研究日志与历史对照](docs/paper/draft-v2.md)
- [质量与时间的计时边界](docs/paper/quality-time.md)
- [已完成结果自动表](docs/paper/results-generated.md)
- [最接近工作的审查](docs/exploration/closest-work-review.md)
- [实验复现入口](docs/exploration/reproduction-map.md)

当前研究尚未证明完整 agent 同时超过强基线的精度与端到端效率。已确认渐进向量计算的局部收益，保留卡片、路由、合成审核等正负结果。500 条原词条件与 49 条语义条件是诊断数据，不替代真实企业需求 gold。

本地冻结编码器可选依赖：`pip install -e '.[local]'`。建议使用隔离环境，不需训练模型。


## 持久化的快速候选检索

已有本地教师语料时，构建并校验经典候选语言模型的精确编译索引：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/teacher_compiled_lm.py
PYTHONPATH=src python3 -m expert_search.compiled_expert_search '机器人运动控制与路径规划' -k 5
```

索引绑定语料快照，从磁盘加载后可重复查询；结果是主题相关性排序，能力支持仍需原文核验。完整真实需求计时将初始化与持久进程中的查询分开，质量结论只使用有标签的公开基准。

该持久化入口现使用浮点上下界筛选与精确前 K 重放，保留参考模型的返回分数和有序结果。完整教师库20条需求×5次配对运行中，前五名和分数全部一致，暖查询均值14.43→6.43ms；这是同一统计模型的执行优化，不提供相关性提升或端到端代理速度结论。复验：`experiments/certified_topk_benchmark.py`。

统一执行路径原型：`OwnedVectorIndex` → `GroundedPipeline`，贯通需求规划、实际查询编码、归属来源检索与固定片段适用性核验。真实21人队列的执行脚本是 `experiments/grounded_pipeline_smoke.py`；约15–18秒的结果仍未满足低延迟目标，语义正确性未独立验证，详见[初稿6.6节](docs/paper/draft-v3.md)。
