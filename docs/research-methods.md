# 问题定义与方法依据

历史研究入口；当前任务定义、证据边界与最新结果以 [初稿 v3](paper/draft-v3.md) 为准。资料不足不作为真实能力负标签。
## 任务

输入企业技术需求 q 和专家集合 E，每个专家包含研究领域、简介、项目及关联论文等证据 D_e。输出有序专家列表与可核查证据。目标是证据支持的技术匹配程度，不等同于学术声望、论文数量或联系意愿。现有数据不能验证专家的实际可用时间、合作意愿及项目交付能力。

## 本地论文阅读

论文优先读取实际存在的 `/Users/zhoudixin/Desktop/Lai/Benchmark/literatures`。三篇全文提取保存在 docs/research，尚未逐篇阅读全文目录内的所有论文。

1. Balog, Fang, de Rijke, Serdyukov, Si. Expertise Retrieval. Foundations and Trends in Information Retrieval, 2012, DOI 10.1561/1500000024。第 3 节区分 expert finding 和 expert profiling；第 4 节讨论评测指标及测试集；第 5 节区分候选人生成模型、文档生成模型、投票及图方法。当前实现采用文档检索后专家聚合的工程基线，并未严格复现生成概率模型。
2. Feng, Yang, Fang. Bridging the Semantic Gap in Industry–Academia Collaboration: A Two-Stage RAG System for Intelligent Expert Recommendation, 2026。本地全文摘要提出多源信息、术语归一化、语义树及从粗到细检索；与当前任务直接相关。本文的向量聚类语义归一化尚未在本项目实现，不能将当前树导航称为该论文的复现。
3. Sun 等. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents. EMNLP 2023。将生成模型用于相关性重排并讨论排列式蒸馏。当前 agent 是带引用的证据查证排序，并未实现 RankGPT 滑窗排列协议或蒸馏。

## PageIndex 的迁移

参考 https://github.com/VectifyAI/PageIndex 的 README：将文档组织成层次索引，再由模型沿树检索和定位来源。原项目关注长文档问答，本项目关注多专家排序。实现中先用词法召回缩小专家集合，再构造“专家→领域/简介/论文→证据叶”的树，模型选择叶子并读取内容，最后生成带证据的排名。

这保留了树导航与来源回溯；未实现全库向量无关的模型树遍历，也没有直接调用 PageIndex 包。候选召回的遗漏无法通过后续排序弥补。每分支前 40 条的展示截断是当前限制，后续应采用查询相关的叶子摘要或分页导航。

## 实验含义

论文标题恢复已知关联专家，可无需额外人工标注运行；但作者归属只是代理标签，同题其他专家也可能相关。指标回答“能否在排除标题对应证据后找回已知作者”，不能回答“企业应选择哪位专家”。因此真实需求实验需单独标注，不把代理准确率移用到企业需求。

当前基线分别为 BM25 最强证据、BM25 前五证据投票、TF-IDF 变体。仍需补充稠密向量检索、混合融合及同候选集 LLM 重排，才能判断树导航相对常见现代检索流程的增益。实验必须比较同一查询子集、排除规则与候选预算，并同时报告速度和模型用量。
