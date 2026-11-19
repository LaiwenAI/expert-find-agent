# Test whether shared-source downweighting helps full-inventory reviewer search

Search the full reviewer inventories on the frozen 122 KDD and 59 SciRepEval panels from the [profile-pooling benchmark](../cof-global-profile-pooling-v1/README.md). The [CoF release](https://github.com/yuz9yuz/CoF) supplies title and abstract queries, reviewer profiles and expert relevance judgments. The published author-to-paper associations establish profile membership. They do not assign personal contributions.

Compute the number of reviewer profiles containing each paper offline. The adjusted score is the original cosine minus 0.025 times the base-2 logarithm of that count. Each doubling subtracts the same amount. Set this constant before the replay without fitting it to these labels. This is a simple specificity heuristic motivated by the [TREC ownership audit](../trec-ownership-metric-audit-v1/README.md).

| Arm | Reviewer score | Order of papers passed to the verifier |
| --- | --- | --- |
| Dense | Best owned paper | Original cosine |
| Equal blend | Equal blend of centroid and best owned paper | Original cosine |
| Adjust reviewer ranking | Equal blend of centroid and best adjusted owned paper | Original cosine |
| Adjust citation sourcing | Original equal blend | Adjusted cosine |
| Adjust both | Equal blend of centroid and best adjusted owned paper | Adjusted cosine |

Every route retrieves ten reviewers and proposes up to three papers per expert. Use the same frozen two-verdict attachment policy on the first five. Rotate execution order across queries. Compare known positives in the top ten and the top five, and whether new positives receive any model-selected citations. Report unjudged recommendations separately, plus source sharing across profiles and duplicated citations within a response. Sharing is an exposure measure. It cannot identify wrong attribution without evidence-level judgments.

Latency covers live query encoding, global retrieval, and attachment on a single warmed request. The timing excludes corpus loading, index setup, transport, and queuing. The experiment uses previously explored CoF panels. Its reviewer-level labels adjudicate neither individual citations nor the evaluation of substantive constraints.

## RTX 5090 results

Slurm job 42694 on ws5 ran both full-inventory panels. It reproduces the dense and equal-blend controls on all decision metrics. The catalog has 14,397 shared papers among 99,594 KDD sources and 6,668 among 73,257 SciRepEval sources.

| Panel | Arm | Known-positive Recall@10 | Known-positive P@5 | Positive and cited P@5 | Judged Top-5 share | Repeated citation slots | Warm retrieval and attachment P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KDD | Dense | 13.64% | 3.77% | 2.13% | 14.75% | 153 | 126.48 ms |
| KDD | Equal blend | 18.60% | 3.93% | 1.97% | 16.72% | 124 | 134.59 ms |
| KDD | Adjust ranking | 19.42% | 3.44% | 1.64% | 15.74% | 92 | 139.87 ms |
| KDD | Adjust citation sourcing | 18.60% | 3.93% | 1.97% | 16.72% | 104 | 143.85 ms |
| KDD | Adjust both | 19.42% | 3.44% | 1.64% | 15.74% | 77 | 145.32 ms |
| SciRepEval | Dense | 50.98% | 16.61% | 8.47% | 35.93% | 13 | 121.74 ms |
| SciRepEval | Equal blend | 53.59% | 17.63% | 7.12% | 41.36% | 8 | 120.73 ms |
| SciRepEval | Adjust ranking | 51.63% | 15.93% | 7.12% | 38.98% | 7 | 120.93 ms |
| SciRepEval | Adjust citation sourcing | 53.59% | 17.63% | 7.12% | 41.36% | 5 | 122.64 ms |
| SciRepEval | Adjust both | 51.63% | 15.93% | 7.12% | 38.98% | 5 | 125.27 ms |

Relative to equal blend, adjusting the ranking gained five and lost three known positives at ten on KDD. One of the five gains reached the top five and none received a citation there. SciRepEval gained four and lost seven at ten. None of the four gains reached the top five. Adjusting both stages reduced repeated citation slots from 124 to 77 on KDD and from 8 to 5 on SciRepEval. Known-positive P@5 fell on both panels, and positive-and-cited P@5 fell or stayed level. The citation-only route preserved all top-five reviewer identities in both panels and changed some attachments without increasing aggregate positive-and-cited P@5.

Under the combined route, 514 of 610 KDD and 180 of 295 SciRepEval top-five slots are unjudged. The corresponding unjudged cited slots are 201 and 53. Source sharing measures redundancy within the snapshot. Judging attribution requires a person-specific evidence assessment. Keep equal blend as the unadjusted comparison, and retain this penalty as a negative control for evidence sourcing. Earlier expert retrieval research also assigns non-Boolean weights to document-person associations. [Mangaravite and Santos, SIGIR 2016](https://doi.org/10.1145/2911451.2914751) · [KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json)

## Reproduce

Use the pinned CoF inputs and open checkpoints from the profile-pooling benchmark. Run on a Slurm CUDA compute node.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.cof_ownership_specificity KDD EMBEDDING RERANKER NEW_KDD_RESULTS
python -m experiments.retrieval.cof_ownership_specificity SciRepEval EMBEDDING RERANKER NEW_SCIREPEVAL_RESULTS
```
