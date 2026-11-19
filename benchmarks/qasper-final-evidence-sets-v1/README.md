# Selecting complete evidence sets from shared QASPER scores

Give each final selector the same 16 or 32 dense candidates and frozen reranker scores. The baseline returns the five highest fused paragraph scores. Other selectors keep its first two passages and fill the next three slots using maximum marginal relevance, discounted lexical coverage, or learned marginal coverage. The learned arm estimates how much a passage increases the best fraction covered among feasible human evidence sets. The ridge features use frozen relevance, residual terms, redundancy, paragraph distance and section identity.

Fit the ridge model on the prior official development fit partition. Choose the coefficient for each selector and depth on the disjoint development calibration partition. A coefficient of zero recovers pointwise selection. No test labels enter either choice. Evaluate on the prior 144-paper [shortlist replication](../qasper-evidence-shortlist-v1/README.md) and 120 hash-selected papers outside both shortlist panels. The broader QASPER test set has been used in prior aggregate studies.

The task supplies the intended full paper and requires five or fewer passages. The hindsight upper bound counts annotated complete sets that fit within the returned slots and are already inside the candidate pool. The anchored bound additionally reserves the first two pointwise passages. These bounds diagnose different constraints on retrieval and selection. Timings distinguish the shared query encoding and reranking prefix from the selector operation on prepared features. Feature construction, cross-paper retrieval, serving and expert ownership are outside the selector timing.

Set-level evidence selection has established precedents. These controls test a small selector around the shared frozen reranker. [SetR, ACL 2025](https://aclanthology.org/2025.acl-long.861/) · [Joint candidate evidence retrieval, NAACL 2021](https://aclanthology.org/2021.naacl-main.363/)

## Calibrated run

All six selector-depth combinations selected a zero weight on the development calibration criterion. Pointwise selection therefore stayed active for every method. The 144-paper replication exactly reproduced both prior pointwise lists for every question. The hindsight bounds expose the remaining potential at fixed GPU work.

| Panel | Depth | Pointwise complete | With two anchors, upper bound | Unconstrained upper bound |
| --- | ---: | ---: | ---: | ---: |
| Replication, 127 evaluable | 16 | 91 | 106 | 106 |
| Replication, 127 evaluable | 32 | 90 | 116 | 118 |
| Additional, 105 evaluable | 16 | 79 | 94 | 95 |
| Additional, 105 evaluable | 32 | 80 | 100 | 101 |

## Forced-positive diagnostic

As a follow-up, use the best positive weight on development for each arm and prefer the largest weight on a tie. This stress test forces a selection change when the calibrated policy would otherwise keep the pointwise output. It reproduces the fit, calibration and every pointwise list, and runs on the same two panels.

| Depth | Pointwise complete / 232 | MMR complete | Lexical coverage complete | Learned gain complete |
| --- | ---: | ---: | ---: | ---: |
| 16 | 170 | 164 | 166 | 168 |
| 32 | 170 | 160 | 167 | 167 |

At depth 32 a complete set could fit for 219 questions, or 216 with both pointwise anchors reserved. The forced methods do not convert that room into improvements. The additional panel loses complete sets under all three methods at both depths. For example, learned selection at depth 32 changes 19 of 264 returned sets and gains one complete set while losing four. Its selector P95 is about 60 microseconds on prepared features. Use the oracle gap to motivate explicit evidence relations at the same budget. The measured heuristics and ridge model remain controls.

## Reproduce

On a Slurm GPU compute node with the [official QASPER data](https://aclanthology.org/2021.naacl-main.365/) and the pinned checkpoints:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.qasper.evidence_sets \
  DATA_DIRECTORY EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON
```

Add `--forced-positive` for the diagnostic. [Aggregate results](results.json) include both runs, frozen coefficients, the development grid and input hashes. The calibrated run records its script hash from before the diagnostic option was added. Raw per-question rows stay local.
