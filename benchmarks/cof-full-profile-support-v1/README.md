# Audit full-profile source depth for retrieved CoF positives

Search the same [full CoF catalogs](../cof-merit-resident-bundles-v1)
with the original 0.6B dense best-source policy. Keep the original
ten nominees per request. For each released positive among those
nominees, sort every owned source in its indexed profile by original
cosine and source ID. The audit uses the released labels only to
select profiles for exhaustive diagnostic scoring. The serving
retriever and verifier never receive the grades.

Run the frozen 0.6B cross-encoder on every distinct query-source
pair belonging to a retrieved positive. Score shared sources once
per query. Use the original task, 1,024-token cap and sign decision.
For this audit, keep the numerical shape fixed at four rows by 1,024
tokens and pad the last batch by repeating its first row. The
fixed-shape control need not yield the exact same decisions as the
guarded dynamic route. Compare every depth slice to the same scored
verdicts so depth is the only intervention.

Report how many known positives have two nonnegative margins within
the first 3, 6, 12 or all indexed sources, split by ranks 1–5 and
6–10. Also report the number with short histories and the number
with zero or one accepted source in the full profile. The difference
between depths 3 and all is an upper limit on recoverable positives
under the same nominee list, verifier decisions and source snapshot.
The remainder could reflect missing evidence, the use of titles and
abstracts, or verifier error. CoF does not verify personal
contributions in these sources. This audit does not evaluate a live
search policy, human relevance of the underlying sources, or quality
on unjudged reviewers.

## Results

Slurm array 45303 ran on an RTX 5090. Both panels reproduced
the dense top-five and top-ten known-positive counts from
the full-catalog experiment. Counts in the four rightmost
columns require two nonnegative source margins.

| Panel | Dense ranks | Known positives | First 3 | First 6 | First 12 | Complete indexed histories |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KDD | 1–5 | 23 | 13 | 13 | 13 | 17 |
| KDD | 6–10 | 10 | 5 | 5 | 6 | 6 |
| SciRepEval | 1–5 | 49 | 25 | 28 | 31 | 31 |
| SciRepEval | 6–10 | 29 | 8 | 11 | 11 | 11 |

Across the two dense top fives, the first three sources
cover 38 of 72 known positives. Twelve sources cover 44,
and every indexed source covers 48. Sixteen of the remaining
24 have exactly one accepted source, and eight have none.
All 72 have at least two catalog sources. The run scored
19,519 distinct query-source pairs, with 31 prompts truncated
under the pinned 1,024-token cap.

The paired depth comparison allows only original cosine ordering
and the same fixed-shape verifier judgments. These previously
explored public panels are diagnostic. The number of additional
positives reached by full scanning bounds possible recovery
under this particular two-source contract, without implying an
online policy can find them at the original budget.

[KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json)

## Reproduce

Use the frozen indexes and checkpoints from the
[full-inventory profile benchmark](../cof-global-profile-pooling-v1).
Run on a Slurm CUDA compute node with a new output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_full_profile_support_audit \
  KDD QWEN06B_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```

Pass `SciRepEval` for the second panel.
