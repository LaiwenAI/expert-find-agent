# Test near-duplicate suppression before citation verification

Reuse the two query panels, complete reviewer inventory, open embedding and reranker checkpoints, and fixed raw centroid–best-source blend from the [profile pooling study](../cof-global-profile-pooling-v1/README.md). All arms generate the same Top-10 ranking and retrieve up to twelve source options for each reviewer. The runner asserts exact agreement of ranked reviewer IDs on every query. For the first five reviewers, each arm passes up to three sources to the same verification and citation pipeline. Citations require two model-positive sources.

- **Raw prefix** takes the first three sources by query cosine.
- **Near-duplicate suppression** keeps the first source and greedily selects subsequent options unless their normalized title equals a previously selected title or their embedding cosine is at least 0.94. Backfill from the original order when too few distinct options remain.
- **Diversity penalty** keeps the first source and then greedily maximizes normalized source-rank relevance minus 0.35 times maximum cosine to a selected source, with identical normalized titles assigned unit similarity. This is a rank-based variant of [maximal marginal relevance](https://doi.org/10.1145/290941.291025).

Final selected sources keep their original cosine order. Compare cited positives, cited judged negatives, and citation coverage with the same expert and source budget. Measure warm sequential processing, including encoding, full retrieval, selection and live evidence checks. The verifier itself sees the original query and document text. Reviewer labels are partial, and citation verification has no source-level human judgments on this benchmark.

## RTX 5090 results

| Panel | Source policy | Cited known positives | Cited judged negatives | Citation coverage | Affected expert–query slots | Complete P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KDD | Raw prefix | 12 | 40 | 42.62% | 0 / 610 | 134.51 ms |
| KDD | Near-duplicate suppression | 10 | 30 | 31.31% | 324 / 610 | 137.77 ms |
| KDD | Diversity penalty | 11 | 40 | 42.46% | 87 / 610 | 134.16 ms |
| SciRepEval | Raw prefix | 21 | 16 | 28.81% | 0 / 295 | 119.84 ms |
| SciRepEval | Near-duplicate suppression | 19 | 13 | 26.78% | 32 / 295 | 126.56 ms |
| SciRepEval | Diversity penalty | 21 | 16 | 28.47% | 13 / 295 | 126.61 ms |

Job 41973 finished 181 requests per arm. Every query kept the same expert ranking across arms. The raw prefix exactly reproduced the earlier positive and cited-positive scores on both panels. Hard duplicate suppression selected different sources in 53.1% of KDD Top-5 slots and reduced coverage by 11.31 percentage points. Both panels delivered two fewer cited positive slots with hard suppression. The diversity penalty changed fewer source sets and recorded no additional cited positives. Paired bootstrap intervals for cited-positive P@5 include zero for every comparison. Because citation judgments are model outputs and most reviewers lack labels, the reduction in citations to judged negatives establishes neither a population error reduction nor a source-level precision gain. [KDD aggregates](kdd.json), [SciRepEval aggregates](scirepeval.json)

Use the same inputs and checkpoints as the profile pooling study inside a Slurm CUDA allocation.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.cof_citation_diversity KDD EMBEDDING RERANKER NEW_KDD_RESULTS
python -m experiments.retrieval.cof_citation_diversity SciRepEval EMBEDDING RERANKER NEW_SCIREPEVAL_RESULTS
```
