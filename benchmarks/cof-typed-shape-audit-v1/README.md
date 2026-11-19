# Audit batch-shape sensitivity in deferred typed verification

Reuse the checkpoints, candidates, queries, thresholds and scheduling rules from the [deferred typed verification comparison](../cof-deferred-typed-v1/README.md). Run immediate and deferred verification with their original dynamic shapes and with both models using request-specific fixed shapes. Each fixed forward uses four rows and a padding length rounded to the next 64 tokens, computed from the longest candidate input for its model. Duplicate rows fill unused GPU slots; discard their scores. Fixed shapes keep the actual query and source tokens intact. Prepare typed prompt tokens for the entire request when computing the common shape, and charge that work to the measured route.

Compare intermediate results only for the same query–source pairs observed in both schedules. At the generic stage distinguish any score drift from a change across the two decision boundaries, zero and three. At the typed stage record changes in the support decision. For queries with dynamic response disagreement, fix only the generic stage and then only the typed stage. These ablations are diagnostic and their denominators are the selected discrepancies. They do not estimate repair rates on unseen requests. Full response equivalence requires identical ordered reviewers and citations.

Measure complete warm processing of both schedules. Rotate the four main arms across queries. Control batches always use the same tokenized evidence and candidate list. The fixed-shape control establishes a common execution protocol and may itself change some model decisions relative to the original dynamic immediate policy.

## RTX 5090 results

| Panel | Shape control | Exact immediate–deferred responses | Generic threshold changes | Typed decision changes | Deferred full P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| KDD | Dynamic at both stages | 117 / 122 | 3 | 7 | 568.90 ms |
| KDD | Fixed at both stages | 122 / 122 | 0 | 0 | 914.81 ms |
| KDD discrepancy subset | Only generic fixed | 1 / 5 | 0 | 7 | — |
| KDD discrepancy subset | Only typed fixed | 4 / 5 | 1 | 0 | — |
| SciRepEval | Dynamic at both stages | 56 / 59 | 0 | 3 | 578.27 ms |
| SciRepEval | Fixed at both stages | 59 / 59 | 0 | 0 | 723.83 ms |
| SciRepEval discrepancy subset | Only generic fixed | 0 / 3 | 0 | 3 | — |
| SciRepEval discrepancy subset | Only typed fixed | 3 / 3 | 0 | 0 | — |

Job 42135 reproduced the eight dynamic disagreements from the earlier comparison. Stabilizing the typed stage alone eliminated seven of them across the two diagnostic subsets. For the remaining KDD disagreement it eliminated observed typed differences, but a generic margin still crossed a decision boundary. Fixing both stages eliminated all within-protocol disagreements and left no decision differences on common observed source pairs.

The new fixed-shape protocol still differed from the original dynamic immediate reference on 12 of 181 queries. Its extra padding and tokenization work raised P95 substantially. It serves as a reproducible comparison protocol for the two schedules, with empirical rather than universal invariance. The run does not establish output-preserving acceleration relative to the original dynamic immediate policy. [KDD aggregates](kdd.json), [SciRepEval aggregates](scirepeval.json)

Use the verified [confirmation inputs and checkpoints](../cof-typed-evidence-v1/README.md) on a Slurm CUDA node.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.cof_typed_shape_audit KDD EMBEDDING RERANKER READER NEW_KDD_RESULTS
python -m experiments.retrieval.cof_typed_shape_audit SciRepEval EMBEDDING RERANKER READER NEW_SCIREPEVAL_RESULTS
```
