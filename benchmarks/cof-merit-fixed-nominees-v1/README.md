# Resident evidence bundles with fixed dense nominees

Use the [full-catalog CoF panels](../cof-merit-resident-bundles-v1)
with the same source snapshots, query exclusions and labels.
Fix the ten nominees and their order to the compact dense
best-source search. For the five candidates sent to the
verifier, compare three source selection policies:

- **Dense:** present the first three owned sources by 0.6B
  similarity, or all sources for shorter profiles.
- **Bundle only:** present up to three sources from the owned
  resident MERIT bundle with the highest query similarity.
  Fill any unused slots from dense source order.
- **Bundle mixed:** preserve the top dense source, use one
  slot for the first distinct bundle source, then fill from
  dense source order. Deduplicate before applying the cap.

All three routes use the same verifier, source cap and
required support count. Each timed route runs from the
query text, including 8B query encoding when it uses bundles.
The shared verifier may stop or recheck at different points
depending on the chosen sources. Report realized calls and
complete warm serial P95 along with the proportion of
identical inputs to the verifier. Use a paired query interval
for the change in known-positive P@5 with model citations.
CoF's sparse reviewer judgments do not independently
assess these citations. This study isolates source selection
under one nominee ranking and one verifier.

## Results

Slurm array 45296 evaluated both panels on an RTX 5090.
All three routes retain the same incoming top-five order
for all 181 requests. Both bundle routes change at least
one candidate's ordered source list per request.

| Panel | Source policy | Known positives with model citations | Warm request P95 |
| --- | --- | ---: | ---: |
| KDD (122 requests) | Dense | 13 / 610 | 135.99 ms |
| KDD | Bundle only | 4 / 610 | 165.06 ms |
| KDD | Bundle mixed | 13 / 610 | 193.70 ms |
| SciRepEval (59 requests) | Dense | 25 / 295 | 130.27 ms |
| SciRepEval | Bundle only | 6 / 295 | 141.18 ms |
| SciRepEval | Bundle mixed | 22 / 295 | 168.34 ms |

For bundle-only versus dense, the query-paired 95% intervals
for the known-positive and cited P@5 change are −2.95 to
−0.33 percentage points on KDD and −10.17 to −3.39 on
SciRepEval. The mixed route yields no gain on the KDD proxy
and its interval spans zero on SciRepEval (−2.37 to +0.34).
Both bundle routes are slower than the compact dense route.
The full-catalog bundle failure therefore cannot be attributed
solely to changed candidate nomination. Source selection alone
reduces this proxy under the same nominee ranking and frozen
verifier. The sparse reviewer labels leave citation quality
and author contribution unverified.

[KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json)

## Reproduce

Use the same Slurm environment and checkpoints as the
[full-catalog MERIT experiment](../cof-merit-resident-bundles-v1).
The panel names are `KDD` and `SciRepEval`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_merit_fixed_nominees \
  KDD MERIT_SCRIPT_DIR QWEN8B_CHECKPOINT MERIT_ADAPTER \
  QWEN06B_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```
