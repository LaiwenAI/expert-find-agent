# 自动生成的实验结果

本文件只读取已有实验产物；没有文件的实验不报告结果。

## SciFact 词法基线

|方法|查询数|nDCG@10|MRR@10|Recall@10|p50 秒|p95 秒|
|---|---:|---:|---:|---:|---:|---:|
|bm25-max|300|0.6605|0.6270|0.7876|0.0192|0.0343|
|tfidf-max|300|0.6527|0.6134|0.7935|0.0188|0.0337|

只包含本地词法查询，不含模型 API。

## SciFact 冻结语义与混合检索

模型：text-embedding-3-small。Query embeddings precomputed. No online latency or end-to-end speedup claim.

- dense：nDCG@10=0.7261，Recall@10=0.8549。
- rrf：nDCG@10=0.7246，Recall@10=0.8346。
- prefix256-refine100：nDCG@10=0.7261，Recall@10=0.8549。
## CMU Gold 单档案版本

词法基线加权配对损失：0.278407，比较对数：1653。

- max-paper：损失 0.254467。
- bounded-max-paper：损失 0.254467。
- mean-profile：损失 0.242088。
- mean-max-50-50：损失 0.235315。

All embeddings precomputed; excludes network and index cost. Fixed 50/50 is exploratory baseline, not tuned on labels. Single profile realization.

## 合成数据筛选

证据优先批次 20 次尝试，保留 4 条。保留标签仍为 silver，完整查询及反事实语义未获充分验证。

## SIGIR 经典候选语言模型与预编译

|方法|Soft P@5|Hard P@5|本地均值 ms|
|---|---:|---:|---:|
|reference-model1|0.3836|0.1753|3.859|
|compiled-model1|0.3836|0.1753|1.089|
|bm25-max|0.3973|0.1973|27.526|

73 个查询，每条三次交替随机顺序；编译前后分数完全一致。质量属于经典模型，编译仅改变执行。
Engineering compilation of a classical language model, not a novel ranker or model training. All usable profiles scored and sorted; quality filters only to released judged candidates. Query tokenization included, offline construction separate. Repeated paired method order randomized per query. Concurrent API worker exists; not isolated hardware or an end-to-end agent SLA. Three repetitions of the same queries are not independent quality samples.

## SIGIR 同候选共享证据实验

|证据|Soft P@5|Hard P@5|调用均值 s|P95 s|平均输入 token|失败|
|---|---:|---:|---:|---:|---:|---:|
|full|0.3699|0.1836|4.166|6.753|48527|1|
|similarity|0.4137|0.1863|2.377|3.203|8263|0|
|coverage|0.3644|0.1671|2.428|3.552|8776|0|

Query bootstrap is exploratory, assumes independent queries, and is not corrected for multiple comparisons. An interval crossing zero does not establish equivalence or noninferiority. Failed calls use coarse fallback and stay in every denominator. Timing excludes query encoding, initialization and rate waiting; report waits separately. Inline cyclic order permits carryover; shared order balances six permutations. Neither is a controlled load test or end-to-end SLA.

## NIPS 经典候选语言模型与预编译

|方法|Soft P@5|Hard P@5|本地均值 ms|
|---|---:|---:|---:|
|reference-model1|0.6000|0.2706|7.783|
|compiled-model1|0.6000|0.2706|1.878|
|bm25-max|0.5118|0.2647|169.819|

34 个查询，每条三次交替随机顺序；编译前后分数完全一致。质量属于经典模型，编译仅改变执行。
Engineering compilation of a classical language model, not a novel ranker or model training. All usable profiles scored and sorted; quality filters only to released judged candidates. Query tokenization included, offline construction separate. Repeated paired method order randomized per query. Concurrent API worker exists; not isolated hardware or an end-to-end agent SLA. Three repetitions of the same queries are not independent quality samples.

## NIPS 同候选共享证据实验

|证据|Soft P@5|Hard P@5|调用均值 s|P95 s|平均输入 token|失败|
|---|---:|---:|---:|---:|---:|---:|
|full|0.5588|0.2824|4.576|7.721|76995|0|
|similarity|0.5176|0.2765|2.201|3.561|7246|0|
|coverage|0.5412|0.2706|2.126|2.832|7626|1|

Query bootstrap is exploratory, assumes independent queries, and is not corrected for multiple comparisons. An interval crossing zero does not establish equivalence or noninferiority. Failed calls use coarse fallback and stay in every denominator. Timing excludes query encoding, initialization and rate waiting; report waits separately. Inline cyclic order permits carryover; shared order balances six permutations. Neither is a controlled load test or end-to-end SLA.

## 完整教师语料的精确编译

96,749 条记录、9,861 位有可用证据教师；20 条原始需求，3 次重复。
查询均值：原始 720.310ms，编译 42.429ms。
Real teacher corpus and original 20 full demand queries; no relevance judgments or quality improvement claim. Exact compilation of classical LM only. Three paired repetitions with randomized method order per query; repetitions are not new independent quality samples. Full-corpus score and sort included; offline normalization, compilation, serialization and loading separate. Other experimental workers may use CPU; not an isolated end-to-end service SLA. Arrays exclude Python/object/text memory.

## 新合成候选与逐条件复审

- 旧词法候选：40 次构造，初审候选 14；不是最终复审通过数。
- 语义候选加领域来源：40 次构造，初审候选 23；不是最终复审通过数。
{"audited": 37, "statuses": {"unresolved": 10, "rejected_audit": 13, "silver_verified": 14}, "models": ["gpt-4o", "gpt-6-luna"], "warning": "Re-audits only prior silver candidates from both cohorts. Same two model families, not an independent human gold check. Complete condition-ID coverage and source/demand quote location are mechanical; semantic entailment and requirement selection remain model judgments. Unresolved is not a negative label."}

## 结论边界

目前未证明新组合方法在强基线之上普遍同时提速提准。经典方法、工程编译、公开数据标签与产业来源支持分别报告；数据候选保持 silver，不混合为总体收益。
