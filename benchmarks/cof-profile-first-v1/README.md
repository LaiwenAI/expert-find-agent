# Profile-first evidence search in the full CoF reviewer inventory

The [previous full-inventory pooling comparison](../cof-global-profile-pooling-v1) found that the mean of the profile centroid and best owned source recovered more labeled reviewers than best-source scoring alone. This experiment isolates the cost of computing that blend. The design follows the classical distinction between profile-based and document-based expert search. Every arm uses the same final score and the same three-source attachment budget per reviewer. The shortlist merely limits the reviewers whose sources receive exact scores. [Classical expert finding models](https://doi.org/10.1145/1148170.1148181)

| Arm | Experts whose sources are scored |
| --- | --- |
| Full blend | All experts |
| Centroid 32 | Top 32 by one normalized profile centroid |
| Two centers 32 | Top 32 by the better cosine of two offline spherical-cluster centers |
| Union 16 / 32 / 64 | Reserve half the slots for each profile ranker, then fill overlaps by centroid rank |

The shortlist routes batch all distinct sources owned by their shortlisted reviewers into one GPU scoring call. Coauthored sources in that batch are scored once. For each reviewed user-facing result, the frozen Qwen3-Reranker-0.6B independently checks the three selected sources using the same procedure as the earlier full-inventory experiment. Published positive labels judge reviewer–paper compatibility. The benchmark has no independent labels for source-level matching or contributions.

The primary comparisons are known-positive Recall@10, known-positive P@5, and labeled-positive P@5 with model-selected citations. We also report sorted expert and source identity parity against exhaustive scoring, the amount of missing judgment, and its effect on paired precision bounds. End-to-end benchmark P95 includes live Qwen3-Embedding-0.6B query encoding, ranking and source attachment on one GPU. Source loading, transport, concurrency and offline index building are outside the request timer.

Both panels and all candidate budgets are fixed before the run. This experiment reuses the earlier public KDD and SciRepEval panels and live model checkpoints. It fits no supervised parameters. Absolute timings characterize this runner, including GPU gathering overhead, rather than a general bound on optimized shortlist designs.

## Results

Slurm job 42765 ran both panels on one RTX 5090 on ws5. Both cohort hashes and all reported exhaustive relevance and citation metrics match the earlier full-blend experiment. Each request used the same three-source attachment policy.

| Panel | Arm | Mean scored sources | Known-positive Recall@10 | Known-positive P@5 | Positive and cited P@5 | Retrieval P95, ms | Retrieval + attachment P95, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KDD (122) | Full blend | 99,594 | 18.60% | 3.93% | 1.97% | 9.42 | 134.78 |
| KDD | Centroid 32 | 4,026 | 16.53% | 3.93% | 1.80% | 9.48 | 133.86 |
| KDD | Two centers 32 | 4,810 | 16.94% | 3.93% | 1.97% | 9.63 | 138.01 |
| KDD | Union 16 | 1,786 | 14.46% | 3.61% | 1.15% | 9.22 | 132.50 |
| KDD | Union 32 | 4,110 | 16.53% | 3.77% | 1.80% | 9.70 | 135.00 |
| KDD | Union 64 | 9,066 | 18.60% | 3.93% | 1.97% | 10.67 | 135.65 |
| SciRepEval (59) | Full blend | 73,257 | 53.59% | 17.63% | 7.12% | 9.08 | 120.95 |
| SciRepEval | Centroid 32 | 3,998 | 43.14% | 15.93% | 7.12% | 9.45 | 116.09 |
| SciRepEval | Two centers 32 | 4,716 | 47.06% | 16.61% | 7.12% | 9.66 | 127.54 |
| SciRepEval | Union 16 | 1,994 | 36.60% | 15.25% | 6.44% | 9.16 | 127.32 |
| SciRepEval | Union 32 | 4,186 | 45.75% | 16.27% | 7.12% | 9.45 | 116.27 |
| SciRepEval | Union 64 | 8,681 | 51.63% | 17.63% | 7.12% | 10.68 | 123.07 |

Union 64 saves 90.9% and 88.2% of source comparisons with the same aggregate labeled-positive P@5 and positive-and-cited P@5. The entire ten-expert sequence and its proposed sources match on 55/122 KDD and 38/59 SciRepEval queries. Returned citations match on 97/122 and 52/59. On KDD, the two added and two lost labeled positives cancel at ten. On SciRepEval, one is added and four are lost. Equal aggregate precision can therefore conceal substituted experts and citations.

All profile-first budgets either miss labeled positives, raise measured P95, or both on at least one panel. Union 64 raises retrieval P95 by 1.25 and 1.60 ms despite its reduced arithmetic. The faster observed complete route on SciRepEval, centroid 32, gives up 16 net known positives at ten. Keep exhaustive GPU scoring as the timing and recall reference for these resident inventories. To justify a shortlist strategy, the next test should compare acceleration of the candidate scan against actual latency under a bigger or concurrent workload, while measuring citation changes. To justify a quality improvement, test source verification and evidence allocation at fixed candidate rankings and add independent evidence judgments.

[KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json)

## Reproduce

Prepare the pinned CoF files and model checkpoints described for [full-inventory profile pooling](../cof-global-profile-pooling-v1). Run a fresh output path on a Slurm CUDA compute node.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_profile_first \
  KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_KDD_RESULTS
PYTHONPATH=src:. python -m experiments.retrieval.cof_profile_first \
  SciRepEval EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_SCIREPEVAL_RESULTS
```
