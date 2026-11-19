# Completing a partially observed evidence set on QASPER

This diagnostic separates candidate pruning from evidence selection. Given the source paper, the question, and some already verified paragraphs, return one new paragraph that completes a human-annotated evidence set. It tests the action after partial evidence is available, complementing the earlier predicted-prefix and citation-edit experiments.

Use the reused QASPER v0.3 official test split and the previous one-question-per-paper multi-evidence cohort. A stable hash chooses a view containing all but one paragraph of a reference supported by extractable text. Exclude any view that already contains a complete known reference. A candidate is a hit if adding it completes any available reference. References are evaluated against extractable paragraphs, with float markers removed. A completing paragraph is guaranteed by construction, so forced selection here gives no estimate of whether to stop or abstain.

Rank unseen paragraphs by binary query-term overlap with paper-local inverse document frequency. The gap variant doubles the weight of query terms absent from all revealed paragraphs. Cross both whole-paper and top-four budgets with question-only and anchor-conditioned prompts. Include gap top four and gap top eight as low-cost alternatives. Anchor-conditioned prompts use up to 128 tokens from each revealed paragraph. All arms reuse the same frozen Qwen3-Reranker-0.6B checkpoint, relevance instruction, passage text, and deterministic tie break. No model or cutoff is fit on these results.

A **shortlist miss** means the model never receives a paragraph that can complete an annotated set. A **selection miss** means such a paragraph is scored but loses to another paragraph. Report paired 95% bootstrap intervals relative to the full anchored arm, with one sampled question per paper. P95 is the reranking component alone, including tokenization and all model batches, after one warmup and randomizing arm order per question. It excludes shortlist construction, corpus-level search and evidence verification. The whole-paper sweep uses batches of at most 16.

## Results

All 137 papers yield an eligible view. Slurm job 42717 ran on one RTX 5090 on ws5. Intervals are descriptive paired differences in completion rate versus full anchored reranking.

| Route | Completed | Reachable | Shortlist misses | Selection misses | Mean scored pairs | Reranking P95, ms | Difference, 95% interval (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full, question only | 31 | 137 | 0 | 106 | 47.18 | 249.45 | −6.57 [−14.60, +1.46] |
| Full, with anchors | 40 | 137 | 0 | 97 | 47.18 | 554.85 | Reference |
| Question overlap top 4, question only | 19 | 36 | 101 | 17 | 4 | 22.59 | −15.33 [−24.09, −6.57] |
| Question overlap top 4, with anchors | 25 | 36 | 101 | 11 | 4 | 39.11 | −10.95 [−18.98, −2.92] |
| Gap overlap top 4, with anchors | 23 | 34 | 103 | 11 | 4 | 39.55 | −12.41 [−21.17, −3.65] |
| Gap overlap top 8, with anchors | 26 | 57 | 80 | 31 | 8 | 77.27 | −10.22 [−18.25, −2.92] |

The four-paragraph lexical cap makes completion impossible in 101 of 137 views. At the same cap, doubling unseen query terms neither improves reachability nor recovers completions. Doubling the gap shortlist exposes a completing paragraph in another 23 views, with a net gain of three completions. The full anchored reranker sees a valid candidate for every question and still misses 97. Providing anchors increases observed full-pool completions by nine, with a paired interval spanning zero. The larger anchored prompts lead to 25 truncated pairs among 6,464 scored by the full anchored arm.

Jointly improving recall and selecting a complementary paragraph is therefore the next diagnostic target. An anchor-conditioned retriever such as [Baleen's condensed retrieval](https://proceedings.neurips.cc/paper_files/paper/2021/hash/e8b1cbd05f6e6a358a81dee52493dd06-Abstract.html) motivates an equal-budget comparison with the lexical gap rule. Evaluate shortlist reachability first, then freeze the shortlist to test a complement-trained selector. These results concern known papers with human-selected anchors on a reused test cohort. Open expert search also needs to find the right source and obtain a reliable first anchor.

[Aggregate results](results.json)

## Reproduce

Use the [official QASPER repository](https://github.com/allenai/qasper-led) for the v0.3 release and a local Qwen/Qwen3-Reranker-0.6B checkpoint with the project's file digest manifest. The runner validates the data and checkpoint hashes and requires Slurm CUDA.

```sh
PYTHONPATH=src:. python -m experiments.qasper.remaining_evidence \
  QASPER_DIRECTORY RERANKER_CHECKPOINT NEW_RESULTS.json
```
