# Confirm methodological evidence during full-corpus reviewer ranking

Use the 122 KDD and 59 SciRepEval panels, full reviewer inventories, and frozen raw centroid–best-source blend from the [profile pooling study](../cof-global-profile-pooling-v1/README.md). Keep the same ten reviewers and up to three owned source candidates per reviewer. Rank confirmed reviewers before unconfirmed ones, preserving initial order within each group. A reviewer needs two accepted sources for confirmation. The shared frontier checks only sources that can change the current ranked output. All routes stabilize generic margins within the existing numerical guard before treating them as decisions.

Use the same open-weight 0.6B reranker in all arms. The generic arm accepts its nonnegative scores. The selective arm sends generic positives below margin 3 to a frozen 4B reader. The all-positive arm confirms every generic positive with that reader. The 4B prompt asks whether the source documents a specific methodological, implementation, or evaluation connection to the query paper. It distinguishes support, contradiction and insufficient evidence. Only support passes. The confirmation stage can suppress a generic positive. Generic negatives never reach the 4B reader. Share query–document judgments within the request; ownership and personal contribution remain separate.

Rotate arm order across queries. Report live embedding, full-corpus retrieval, prompt construction, both model stages, and ranking. Timing excludes serving transport and index build. Released hard labels evaluate reviewer relevance only. A citation to a labeled reviewer is a model output and remains unjudged at the citation level.

## RTX 5090 results

| Panel | Confirmation | Known-positive P@5 | Cited known positives | Cited judged negatives | Total cited slots | Full P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KDD | Generic only | 3.77% | 15 | 44 | 315 | 262.54 ms |
| KDD | Selective typed | 3.93% | 8 | 21 | 136 | 612.78 ms |
| KDD | All generic positives | 3.77% | 2 | 11 | 63 | 776.30 ms |
| SciRepEval | Generic only | 17.63% | 27 | 26 | 123 | 263.72 ms |
| SciRepEval | Selective typed | 18.98% | 18 | 8 | 44 | 589.15 ms |
| SciRepEval | All generic positives | 17.97% | 8 | 4 | 24 | 750.31 ms |

Job 42090 saw the same ten candidate IDs and order in all arms for 181 queries. Selective confirmation decreased cited known positives on both panels. The query-paired 95% intervals for cited known-positive P@5 change were −2.13 to −0.33 percentage points on KDD and −5.08 to −1.36 points on SciRepEval. The one-sided rejection stage called the 4B reader 2.66 and 2.85 times per request on average, and rejected 3.57 and 3.73 generic-positive sources. It more than doubled full P95 on both panels.

The lower cited-negative counts came alongside sharply lower citation coverage. This benchmark does not show a gain at equal coverage or a citation-level risk reduction. The typed stage only rejects generic positives and cannot recover a relevant source rejected by the first scorer. [KDD aggregates](kdd.json), [SciRepEval aggregates](scirepeval.json)

Use the same CoF snapshots and open checkpoints as the linked pooling study, plus the [verified Qwen3-4B checkpoint](../nli4ct-adaptive-reader-v1/README.md). Run inside a Slurm CUDA allocation.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.cof_typed_evidence KDD EMBEDDING RERANKER READER NEW_KDD_RESULTS
python -m experiments.retrieval.cof_typed_evidence SciRepEval EMBEDDING RERANKER READER NEW_SCIREPEVAL_RESULTS
```
