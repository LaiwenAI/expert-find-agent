# Selecting numeric witnesses in mixed bilingual passages

Six new synthetic scenarios generate seven pairs of sentences per language. Each pair is tested in both orders. The 168 views cover proposed targets, another setting, another metric, agreeing measurements, conflicting measurements, an unavailable measurement, and compatible units. The two controlled sentences retain explicit equipment, measurement status, and operating conditions. The model sees their text. Construction-time relevance and outcome labels are used only for scoring.

The direct reader predicts reject, eligible, or unknown from the whole passage. The joint pointer selects neither sentence, the first, the second, or both in one 4B forward. Independent scope verification makes one 4B decision per sentence without seeing the numerical limit. Exact decimal comparison then uses quantities in the selected original sentences. When no sentence qualifies or selected measurements give opposing threshold decisions, the factored route returns unknown. The agreement control accepts a sentence set only when the two selectors agree.

| Route | English correct / 84 | Chinese correct / 84 | English false approvals | Chinese false approvals | Chinese P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct reader | 56 | 67 | 12 | 13 | 14.66 ms |
| Joint pointer | 69 | 59 | 9 | 6 | 14.60 ms |
| Independent sentence checks | 80 | 80 | 4 | 4 | 22.26 ms |
| Selector agreement | 64 | 50 | 3 | 3 | 36.86 ms |
| Provided sentence set | 84 | 84 | 0 | 0 | — |

Independent checks preserve all 36 valid approvals in each language. They abstain on all 12 conflicting-result views per language. Their four errors per language occur when neither sentence contains a comparable empirical measurement. Results agree with the expected decision in both sentence orders for 40 of 42 groups per language. The more expensive selector agreement lowers false approvals while losing 10 valid approvals in English and 16 in Chinese, relative to independent checks.

Job `40621` uses the frozen Qwen3-4B-Instruct-2507 checkpoint on RTX 5090. P95 includes prompt preparation, sequential forwards and comparison. This controlled sentence-level evaluation supplies the sentence boundaries. Unsegmented text, distributions of real conflicting measurements, and contributor attribution require separate validation. [Aggregate results](results.json)

Run [the replay](../../experiments/transfer/bilingual_numeric_passages.py) with `SCENARIOS CHECKPOINT NEW_RESULT_JSON` on a Slurm CUDA compute node.
