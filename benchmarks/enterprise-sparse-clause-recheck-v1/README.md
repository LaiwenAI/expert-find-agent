# Recheck only the clauses that trigger qualification fallback

The [cross-precision qualification parser](../enterprise-cross-precision-v1)
uses a BF16 cached pass and an FP16 check for low-margin blocks.
When the two precisions disagree on a prior qualification or an FP16
qualification margin is within 0.5 logits, the existing route rechecks
every clause in the block with independent, full-context BF16.

Keep its block trigger unchanged. In the candidate routes, recheck
only rows with a qualification disagreement or an FP16 qualification
margin within the same 0.5-logit threshold. Retain the full original
request context per selected row. Preserve cached BF16 scores for
the other rows. Compare dynamically shortened padding and the
original block width with the same selected rows. Fixing width
preserves original position IDs and padding geometry per row while
still changing the number of rows in the forward. This is an
empirical omission rule. Agreement between two precisions does not
bound errors from omitted rows.

A follow-up keeps the original block width and compares margins of
2.0 and 4.0 logits in either precision, in addition to qualification
disagreement. These wider bands apply only after the existing block
trigger. They do not add new block checks. Diagnose each changed
qualification mark against the independently rechecked block.
Distinguish changes in skipped rows from changes in forwarded rows.
The follow-up records these comparisons in every arm. Compare
timing only within each experiment.

The service path is the existing guarded enterprise search. All
variants interpret the same 80 original development requests,
retrieve from 52,667 sources, verify the same fallback policy and
compare exact qualification offsets, ranked experts, citations,
provisional decisions, and unresolved states. Validate the
instrumented whole-block reference against the unchanged parser
for each request. Report total and parser latency, fallback rows and
executed fallback tensor slots, and paired bootstrap intervals.
Repeat with reversed arm order. Also check the 72 anchored bilingual
[public heading controls](../enterprise-heading-roles-v1).
No sealed request or human enterprise judgment enters this comparison.
Results are aggregate only.

## Results

### Original row selection

Both tasks in Slurm array 46412 complete on an RTX 5090 on ws3.
The two orders reproduce the same output changes, work counts
and public control results.

| Route | Fallback rows | Fallback tensor slots | Live P95 (ms) | Changed qualification sets / 80 | Issued requests / 80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole block | 56 | 74,680 | 1053.9–1060.0 | 0 | 37 |
| Selected rows | 7 | 9,250 | 744.6–748.8 | 4 | 36 |
| Selected rows, original width | 7 | 9,335 | 744.2–749.1 | 2 | 37 |

Both sparse routes change exact primary decisions. Shortening the
width also changes a fallback output. Neither route preserves the
complete search result. On every arm, public heading
controls give 60/72 correct, two false qualification marks, and ten
missed qualification marks. Those aggregate scores do not expose
the private development disagreements.

### Wider guard at the original padding width

Both tasks in Slurm array 46418 reproduce the diagnosis on ws3.
Counts refer to one pass of the 80 original development requests.

| Route | Fallback rows | Fallback tensor slots | Live P95 (ms) | Changed qualification sets | Changed marks on skipped / forwarded rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole block | 56 | 74,680 | 1061.6–1078.3 | 0 | 0 / 0 |
| Original selection, fixed width | 7 | 9,335 | 750.0–762.4 | 2 | 0 / 3 |
| Either-precision margin ≤2.0 | 7 | 9,335 | 752.4–763.2 | 2 | 0 / 3 |
| Either-precision margin ≤4.0 | 9 | 11,265 | 756.8–764.7 | 3 | 0 / 4 |

Regardless of guard width, the observed qualification changes occur
in rows that actually run through the reduced batch. The four-logit
arm changes one primary issuance and one recovery decision. Simply
selecting more borderline clauses does not fix the differences
introduced by the changed batch height. This experiment does not
locate the precise arithmetic operation that changes. Keep the
canonical whole-block geometry as the evidence-preserving baseline.
Timing is for warmed serial execution without transport or queueing.
The development comparisons use model outputs, and public control
scores do not establish correctness on enterprise demands.

[Aggregate runs and paired intervals](results.json)

## Reproduce

Restore the [guarded enterprise inputs](../enterprise-packed-role-load-v1#reproduce)
and use a Slurm CUDA allocation. Repeat with `--reverse`. For the
original timing without row attribution, use the
[original runner](https://github.com/LaiwenAI/expert-find-agent/blob/80fbc6ae85937475a236da1d6d9506444b24cd47/experiments/retrieval/enterprise_sparse_clause_recheck.py).
Use the current runner with `--guarded-rows` for the follow-up.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_sparse_clause_recheck \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS PRIVATE_ROLE_WINDOWS PRIVATE_VECTORS.npy \
  PRIVATE_VECTOR_META MODELS benchmarks/enterprise-packed-role-search-v1/results.json \
  benchmarks/enterprise-heading-roles-v1/controls.json NEW_RESULTS.json
```
