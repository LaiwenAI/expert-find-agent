# Source-first recall at a fixed complete-evidence budget

In the [predicted-anchor test](../qasper-predicted-completion-v1), the six-paragraph flat candidate window rarely contained a complete set of annotated evidence. This experiment measures whether using document metadata before reading passages changes that ceiling. Use the same reused official QASPER v0.3 development and test panels, all papers within each split, and frozen Qwen3-Embedding-0.6B. Use the original question or append the same deterministic short title cue. No arm fits on the new outcomes.

| Ranker | Construction of the six-passage window |
| --- | --- |
| Flat | Rank all passages by the question embedding |
| Source one | Rank title-plus-abstract vectors, then use the six highest-ranked passages from the best paper |
| Balanced | Retrieve the three best papers by the same metadata similarity and take passages in round-robin order, taking one passage from each source before taking a second |
| Title gate | In the cued condition, use a unique normalized title n-gram match, or use source one when there is no unique match. Rank passages within the selected paper by the same dense score |

Passages share the same section-plus-text index. The flat and metadata scores use a common question vector per condition. All rankers start with the top paragraph, allow two further outputs, and expose at most four remaining candidates at each step. Compare taking the next ranked passage and reusing the fixed learned complement model. No new training or reranker calls are needed.

A **complete window** contains every paragraph of any extractable annotation set. An **attainable** window must also allow this set to fit in three outputs while preserving the rank-one seed. A completion meets the same annotation with the three actual outputs. Title cues are drawn verbatim from the target title, which makes exact metadata lookup a strong control. A query without this cue can refer to a designated article through implicit context, so raw-query evaluation is harder and some answers may remain ambiguous.

Score counts include source vectors where used. The component latency includes live query encoding, ranking, and selection. It excludes indexing, serving and final attribution. The timing uses a resident array, including a full CPU dense matrix multiply for flat search, and does not establish the relative latency against ANN or optimized GPU retrieval at other scales.

## Results

Slurm job 42756 completed on one RTX 5090 on ws5. It reproduces the earlier flat route's attainable and completion counts. The table reports attainable windows, completion under the fixed learned selector, mean comparisons of vectors, and component P95.

| Panel / query | Route | Attainable | Completed | Comparisons | P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Dev raw (112) | Flat | 7 | 5 | 13,194 | 16.59 |
| Dev raw | Source one | 6 | 3 | 328 | 8.98 |
| Dev raw | Balanced | 1 | 1 | 424 | 13.14 |
| Test raw (137) | Flat | 5 | 1 | 19,710 | 17.45 |
| Test raw | Source one | 7 | 2 | 464 | 9.50 |
| Test raw | Balanced | 2 | 1 | 565 | 12.21 |
| Dev cued (112) | Flat | 10 | 8 | 13,194 | 14.90 |
| Dev cued | Source one | 14 | 9 | 329 | 9.03 |
| Dev cued | Balanced | 3 | 3 | 423 | 12.81 |
| Dev cued | Title gate | 17 | 12 | 328 | 9.10 |
| Test cued (137) | Flat | 7 | 2 | 19,710 | 16.72 |
| Test cued | Source one | 12 | 4 | 464 | 9.60 |
| Test cued | Balanced | 2 | 2 | 555 | 12.24 |
| Test cued | Title gate | 19 | 7 | 465 | 9.62 |

For test questions with an explicit title fragment, title gating selected the correct source in 127 cases, compared with 73 for flat retrieval. Attainable completions rose from 7 to 19, and actual completions from 2 to 7 (paired 95% interval +0.73 to +7.30 percentage points). The title route scored about 42 times fewer vectors and cut component P95 by 42.5%. The gains use literal title cues. The metadata index added 1.7 MB and took 1.43 seconds to build for the test collection.

For raw questions, source-one routing trades source coverage for more depth within its chosen paper. The target paper appears anywhere in the six-passage test window in 59 flat and 29 source-one outputs, yet full references can fit in five and seven windows, respectively. The net completion changes reverse direction across the two panels.

Spreading the window evenly does increase source coverage compared with source-one routing, from 91 to 103 test windows in the cued condition. It simultaneously reduces attainable evidence sets from 12 to 2. The quota exposes at most two passages per paper when all three sources have enough passages, which excludes references of three or more paragraphs by design. Evaluate source allocation with both source recall and whole-evidence recall. The next expert-search comparison needs to preserve room for multiple witnesses when a candidate already has credible support and retain an alternate-source path when the source signal is weak.

[Aggregate results and dense-selector control](results.json)

## Reproduce

Use the public QASPER v0.3 files, the verified embedding checkpoint, and the published weights from the [completion selector experiment](../qasper-completion-selector-v1).

```sh
PYTHONPATH=src:. python -m experiments.qasper.source_first_completion \
  QASPER_DIRECTORY EMBEDDING_CHECKPOINT \
  benchmarks/qasper-completion-selector-v1/results.json NEW_RESULTS.json
```
