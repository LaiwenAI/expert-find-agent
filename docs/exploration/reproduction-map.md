# 研究复现入口

所有命令在项目根目录执行，不需要模型训练。

| 产物 | 命令或模块 | 前提 |
|---|---|---|
| 作者恢复基线 | python -m expert_search benchmark | corpus、SQLite 索引 |
| SciFact 词法 | python -m expert_search.research_cli public-lexical | 下载并适配数据 |
| SciFact dense/RRF/渐进质量 | python -m expert_search.research_cli public-hybrid | API 语料和查询向量 |
| 渐进计算延迟 | python experiments/progressive_latency.py | 已编码向量，BLAS 单线程 |
| CMU 词法 | python -m expert_search.research_cli cmu-lexical | 固定上游 d_20_1 |
| CMU 冻结画像 | python -m expert_search.research_cli cmu-dense | CMU API 向量 |
| CMU 卡片 | python -m expert_search.card_benchmark | 已生成和编码的卡片 |
| CMU 在线证据 | cmu_evidence_benchmark.run(limit=58) | 已配置模型服务 |
| 模型同子集比较 | python -m expert_search.cmu_compare | 记录的请求结果 |
| 条件任务构造 | python -m expert_search.controlled_data | 本地教师语料 |
| 条件任务评测 | python -m expert_search.controlled_benchmark | 已构造数据和快速索引 |
| 本地 BGE | python -m expert_search.local_encoder_benchmark | 可选 local 依赖、下载冻结模型 |
| 结果表同步 | python -m expert_search.research_report | 已完成的实验文件 |

命令需设置 `PYTHONPATH=src`。本地编码计时使用 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，模型 ONNX threads=1。CPU、服务和数据规模改变后必须重测。

数据获取及授权说明见 public-benchmark.md。模型提示与缓存版本在源码中；模型输出、失败和哈希在 artifacts 下。该目录包括第三方数据，不应整体加入开源仓库。原文 PDF 提取文本与教师资料也不是本项目代码许可的一部分。


## 新增复现入口

|产物|命令|边界|
|---|---|---|
|CoF 经典语言模型|python -m expert_search.cof_language_benchmark --dataset NIPS|全部画像评分，已判定集合评价|
|精确编译配对计时|python experiments/compiled_lm_benchmark.py|SIGIR/NIPS 顺序运行，三个重复不算新质量样本|
|完整教师索引|python experiments/teacher_compiled_lm.py|含保存、加载及三次随机配对；无产业精度金标|
|持久化检索|python -m expert_search.compiled_expert_search '机器人运动控制'|初始化与查询单独报告，统计主题相关性|
|共享证据统计|python -m expert_search.packing_analysis --dataset NIPS --serialization shared|完整性、失败回退、配对差值、P95|
|本地集合聚合|python -m expert_search.cof_late_interaction --dataset NIPS|需 local 环境；既有 MaxSim 组件，不是原创声明|
|语义合成候选|python -m expert_search.semantic_synthesis_pool|local 环境；准备阶段无模型 API|
|候选生成审核|python -m expert_search.semantic_synthesis_pool --execute|真实模型调用，来源和任务哈希检查|
|逐条件复审|python -m expert_search.scoped_dataset_audit|完整条件 ID、唯一源引用和需求引用，仍是 silver|
|复审导出|python -m expert_search.scoped_dataset_audit --export|重验模型输出及源产物，不导出原文|

冻结本地模型命令使用 `.venv-local/bin/python` 或按项目 local 可选依赖安装的环境。所有计时命令都需显式设置单线程环境；并行运行共享 CPU 的计时任务会改变测量条件。
