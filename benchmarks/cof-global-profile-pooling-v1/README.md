# Fixed profile pooling in full-inventory CoF reviewer search

The [CMU and MERIT results](../cmu-profile-ranking-v1/README.md) identified fixed dense aggregation as a strong baseline under supplied reviewer profiles. This experiment tests the same idea in full-inventory retrieval, using the public CoF source and reviewer snapshots with their released hard judgments. Every arm searches the entire reviewer inventory. KDD uses the frozen 122-query panel from the [global candidate audit](../cof-global-candidate-audit-v1/README.md). SciRepEval excludes queries matching an ownership record by ID or title through the existing vector loader, and then excludes normalized titles in the NIPS judgments. The panels and ranker equations are fixed before this comparison.

| Ranker | Score used to order reviewers |
| --- | --- |
| Best source | Maximum cosine of an owned source |
| Profile centroid | Cosine of the normalized mean of all owned source vectors |
| Best-three mean | Mean cosine of the best available three owned sources |
| Equal blend | Equal mean of profile centroid and best-source scores |
| Two centers | Higher cosine of two offline spherical clusters in the profile |
| Two centers plus best source | Equal mean of the two-center and best-source scores |

The two-center initializer chooses the source furthest from the profile centroid and then the source furthest from that seed, using stable source order for ties. Six assignment updates use normalized cluster means. Single-source reviewers reuse their only source as both centers. Pure centroid routes rank reviewers from the profile vectors, then perform exact source scoring only for the ten selected reviewers. The source-based routes score the entire source array. All routes propose up to three sources per reviewer, using owned-source cosine order.

The frozen Qwen3-Embedding-0.6B encodes each query live. The shared attachment policy uses the same frozen Qwen3-Reranker-0.6B to check candidate sources. It requires two model-supported sources for a cited recommendation and preserves reviewer rank. Report known-positive retrieval separately from the rate of known positives accompanied by model-selected citations. CoF does not independently judge those citations. Warm P95 includes encoding, ranking, and when reported, attachment. Index loading, offline clustering, transport and serving under load are outside the timer.

The paired known-positive P@5 interval resamples queries. Partial-label bounds use identical labels for shared reviewer slots and permit the unjudged reviewers gained or lost by a route to be relevant or irrelevant. The resulting range is a limit imposed by missing labels on this panel. It is distinct from the bootstrap interval.

## RTX 5090 results

Job 41607 completed both panels. KDD uses 122 queries, 737 reviewers, 99,594 sources and 242 known positives. SciRepEval uses 59 queries, 665 reviewers, 73,257 sources and 153 known positives.

| Dataset | Ranker | Known-positive Recall@10 | Known-positive P@5 | Positive and cited P@5 | Retrieval P95 | Complete P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KDD | Best source | 13.64% | 3.77% | 2.13% | 10.04 ms | 126.32 ms |
| KDD | Profile centroid | 14.46% | 3.61% | 0.82% | 9.09 ms | 145.45 ms |
| KDD | Best-three mean | 16.53% | 2.95% | 1.64% | 22.63 ms | 144.03 ms |
| KDD | Equal blend | 18.60% | 3.93% | 1.97% | 9.73 ms | 134.57 ms |
| KDD | Two centers | 15.70% | 3.44% | 0.82% | 9.38 ms | 138.99 ms |
| KDD | Two centers plus best source | 18.18% | 3.11% | 1.31% | 9.80 ms | 142.40 ms |
| SciRepEval | Best source | 50.98% | 16.61% | 8.47% | 8.77 ms | 121.36 ms |
| SciRepEval | Profile centroid | 35.29% | 12.88% | 5.08% | 8.41 ms | 122.96 ms |
| SciRepEval | Best-three mean | 54.25% | 18.98% | 9.83% | 17.82 ms | 127.30 ms |
| SciRepEval | Equal blend | 53.59% | 17.63% | 7.12% | 8.51 ms | 120.86 ms |
| SciRepEval | Two centers | 39.87% | 14.24% | 6.44% | 8.66 ms | 123.42 ms |
| SciRepEval | Two centers plus best source | 54.25% | 18.64% | 8.47% | 8.72 ms | 119.59 ms |

The equal blend adds 17 and loses five KDD positives at ten, and adds 14 and loses ten SciRepEval positives. Retrieval P95 stays below 10 ms in both panels. Its improvement at five is smaller and positive-and-cited P@5 drops in both panels. KDD recovers 12 net new known positives but returns one more in the top five. Pure centroids cut global source scoring yet increase complete P95 in both workloads. Two centers preserve more known positives than one center alone, while the blended two-center model trails the equal one-center blend on KDD.

Only 14.1–16.7% of KDD top-five positions and 35.3–43.4% of SciRepEval positions are judged across arms. Thus the recall results describe recovery of released positives. The missing judgments leave substantial uncertainty about full-population precision. [KDD aggregates](kdd.json), [SciRepEval aggregates](scirepeval.json)

## Reproduce

Use the pinned inputs and verified open-weight checkpoints described in the [global candidate audit](../cof-global-candidate-audit-v1/README.md). Run on a Slurm CUDA compute node with both public CoF panels available.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.cof_global_profile_pooling KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_KDD_RESULTS
python -m experiments.retrieval.cof_global_profile_pooling SciRepEval EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_SCIREPEVAL_RESULTS
```
