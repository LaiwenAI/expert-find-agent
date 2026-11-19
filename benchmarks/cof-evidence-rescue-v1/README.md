# Where CoF shortlist gains are lost before citation delivery

The [full-candidate profile comparison](../cof-global-profile-pooling-v1/README.md) found that blending profile centroid and best-source similarity increased recovery of known positive experts at ten. This experiment freezes both released query panels, the expert ranking and the two-source citation rule. It separates candidate position, supplied source depth and model acceptance. It also checks whether the agent can use the additional candidates without increasing the evidence scoring cap.

| Route | Search | Evidence candidates | Output order |
| --- | --- | --- | --- |
| Dense attach three | Best source | Three per original Top-5 expert | Original order |
| Blend attach three | Equal blend | Three per original Top-5 expert | Original order |
| Blend attach six | Equal blend | Six per original Top-5 expert | Original order |
| Blend promote three | Equal blend | Three per Top-10 expert, no hard score cap | Promote experts with two accepted sources |
| Blend budget six | Equal blend | Six per Top-10 expert, at most 15 scored or rechecked source pairs per request | Promote confirmed experts; return the conservative output at the cap |

The budgeted arm computes optimistic and pessimistic outputs over unresolved sources. It checks only sources in the optimistic top five and reuses a query–source verdict across the source's recorded owners. Margins within the guard band stay provisional until rechecked. If the cap is reached before the outputs agree, unidentified sources cannot count as support. Fifteen matches the maximum number of distinct source candidates exposed to the original attachment depth before any rechecks. The budget counts actual unique scoring events including rechecks, so model calls, aggregate tokens and latency can differ even at identical source counts.

For diagnosis only, score up to six sources for every released positive in the blended Top-10 with the same frozen model in batches of four. Label each candidate as shared with the dense Top-10 or added by blending, and distinguish ranks 1–5 from ranks 6–10. This oracle selection is never used by a serving route. It measures model-accepted sources in the retrieved prefix, so a low count can reflect missing relevant evidence, model error, or both. CoF lacks adjudicated citations for these outputs.

The live arms run separately on each query, rotating order and including encoding, ranking and citation delivery. The timer excludes serving transport and the separate diagnostic scores. Known-positive and negative metrics use the released partial reviewer labels. Report citations to unjudged reviewers as unknown rather than treating them as good or bad.

## RTX 5090 results

Slurm job 41619 reproduces the baseline citation counts on 122 KDD and 59 SciRepEval queries. Counts are reviewer–query slots. Checks include rechecks and exclude padding copies.

| Dataset | Route | Known positives with citations | Judged negatives with citations | Mean checks incl. rechecks | Complete P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| KDD | Dense attach three | 13 | 39 | 9.60 | 125.83 ms |
| KDD | Blend attach three | 12 | 40 | 10.04 | 134.20 ms |
| KDD | Blend attach six | 12 | 45 | 19.09 | 246.66 ms |
| KDD | Blend promote three | 15 | 44 | 18.03 | 233.38 ms |
| KDD | Blend budget six | 12 | 37 | 13.81 | 166.36 ms |
| SciRepEval | Dense attach three | 25 | 21 | 12.00 | 121.38 ms |
| SciRepEval | Blend attach three | 21 | 16 | 12.10 | 120.74 ms |
| SciRepEval | Blend attach six | 27 | 22 | 23.75 | 235.45 ms |
| SciRepEval | Blend promote three | 27 | 26 | 21.73 | 200.51 ms |
| SciRepEval | Blend budget six | 24 | 15 | 14.78 | 137.96 ms |

In KDD, 12 of the 17 new positives sit at ranks 6–10. None of those 12 reaches two accepted sources even when the diagnostic reads six. Five have no accepted source and seven have one. In SciRepEval, eight of the 14 new positives sit at those ranks; three reach two accepted sources at either depth. The six-source diagnostic widens support mostly for already shared, higher-ranked candidates. All diagnostic positives have at least two available sources in the six-source window.

Uncapped promotion recovers three extra cited positives on KDD and six on SciRepEval compared with blend attachment at depth three, with higher P95 (134.20 to 233.38 ms and 120.74 to 200.51 ms). It also cites more explicitly judged negative reviewers in this partial label pool. Under the 15-check cap, the agent returns 12 and 24 cited positives, versus 13 and 25 with the best-source attachment control. Complete P95 increases. The rank-and-citation certificate closes within budget on 26/122 KDD and 5/59 SciRepEval requests. The remaining responses retain verified citations only and preserve fallback positions without citations where verification is incomplete. [KDD aggregates](kdd.json), [SciRepEval aggregates](scirepeval.json)

## Reproduce

Use verified checkpoints and the public source and vector snapshots from the linked [profile benchmark](../cof-global-profile-pooling-v1/README.md). Run with a Slurm CUDA allocation.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.cof_evidence_rescue KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_KDD_RESULTS
python -m experiments.retrieval.cof_evidence_rescue SciRepEval EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_SCIREPEVAL_RESULTS
```
