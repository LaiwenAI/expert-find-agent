# MERIT公开专业匹配测试适配

本适配只用于**论文—审稿人专业匹配**，不能替代企业需求迁移或项目成功的标签。[MERIT（EMNLP 2026）](https://aclanthology.org/2026.emnlp-main.2227/)在论文4.1节说明：LR-Bench专业度来自1–5分人工自评，4/5为正，其余为负；适配固定作者公开的610对测试数据，而非模型生成银标。

官方代码固定提交`5fad0ea06148c3c8817aac841de46c98a75eb40c`，来源[MERIT仓库](https://github.com/Luli3220/MERIT)。选择性下载9个文件，共20,021,659字节，逐项校验Git blob SHA-1、文件长度及SHA-256；清单见`docs/exploration/merit-release-assets-20261117.json`。数据与作者代码存放在`data/public/merit-release-20261117`，不随本仓库MIT源码包再分发。

610对中正例254、负例356，规范化标题对应509个不同目标论文。作者资料只提供论文列表，不能据此保证审稿人身份独立；精确标题分组也不等于语义近重复排除。作者的人工自评比本地同模型银标独立，但仍是特定任务的自评标准，不证明未来企业项目能力。

## 冻结比较

所有臂使用`gpt-6-luna`，不训练任何模型，也不运行MERIT已训练的4B/8B权重：

1. `official-long`：直接读取作者发布的完整系统/用户提示，生成要求、证据、总结及最终标签，上限4096完成token。
2. `short-chat`：保留同一核心加补充技能的判定要求及全部目标/资料文本，只返回严格schema下的短标签，上限128完成token。
3. `native-choice`：同一短判定要求和完整资料，用原生Decisions返回二选一结果。

这是实际提示、生成量及接口组合的对照，不能称为精确复现MERIT训练方法，也不是两个接口内部计算完全相同的实验。按原始610顺序循环六种调用排列、三并发、无重试。目标标题/摘要/引言及候选全部论文标题/摘要保留，不沿用作者本地评测脚本的5K prompt截断/过滤。分数只在推理后读入评价；推理case/payload不含`score`或人工标签。

## 失败与评价

API错误、缺最终字段、格式不符保持`label=null`，不补0。分析器必须检查610×3个终态文件全部存在，缺任何一个即拒绝输出准确率报告。报告有效部分的precision/recall/F1/accuracy、覆盖率，以及包含失败项的全队列最坏/最好识别区间；区间不是置信区间。比较另在共同有效配对上按目标论文标题分组bootstrap，时延使用所有实际尝试，包含错误等待。分组不能覆盖未知审稿人身份相关性。不能保证预训练未见这些公开资料或标签。

```bash
python experiments/fetch_merit_release_assets.py
PYTHONPATH=src python experiments/merit_suitability_comparison.py --prepare-only
PYTHONPATH=src python experiments/merit_suitability_comparison.py
PYTHONPATH=src python experiments/analyze_merit_suitability.py
```

推理命令产生付费API调用；已有结果按冻结输入摘要复用。完整610对及1830次请求已全部有效完成：完整推理/短Chat/原生Decisions准确率分别73.28%/68.69%/71.15%，请求均值3.84/1.64/0.91秒。原生相对完整推理的准确率差为-2.13个百分点，论文分组bootstrap区间[-5.51,1.32]；跨零不能证明等效。它是成本与质量权衡，不能称为精度提升。原始企业100条需求的20条封存集未参与本公开辅助任务。
