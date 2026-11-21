# Allocate two recovery windows to the same expert

This live development comparison uses 80 original requests, 52,667
sources and 5,734 author-associated role windows. All methods share the
same frozen parser, dense initial retrieval and packed verifier.
Only initially abstained requests with more than one criterion are
eligible for a different recovery choice, and only when the original
method already allocates two windows.

The baseline checks the two windows chosen by global lexical role
recovery. The alternative independently ranks role windows for each
criterion with the frozen embedding model and selects a pair associated
with one expert. It minimizes the worst criterion rank, then total rank.
Each arm consumes the same number of logical verifier pairs per request.
The verifier tests every criterion against both allotted windows. When
one window supports all criteria, both variants use that window.
Otherwise, the guarded variant also requires each accepted criterion to
have a supporting window independently ranked in its top three. The two
variants revert to baseline recovery if no eligible pair is found.

Run at depth 24, then widen to 256 as an exploratory check. A third
configuration at depth 256 permits two non-overlapping windows in the
same profile field. The citation keeps the field ID and exact offsets.
Two spans of one field do not establish independent corroboration or
distinct underlying projects. The field-level inventory associates 1,098
experts with role windows. Four have windows from multiple fields and
964 have at least two non-overlapping windows within a single field.

## Results

Of 80 requests, the initial stage issued recommendations for 34.
The baseline recovered an additional three. Twenty-three abstained
requests had multiple criteria and a full two-window recovery budget.

| Allocation constraint | Depth per criterion | Pairs allocated (of 23) | Requests issued, open combination | Requests issued, rank-guarded |
| --- | ---: | ---: | ---: | ---: |
| Distinct fields | 24 | 0 | 37 | 37 |
| Distinct fields | 256 | 1 | 37 | 37 |
| Non-overlapping spans allowed | 256 | 23 | 39 | 37 |

The two distinct-field configurations preserve every issued candidate
and exact citation. Permitting disjoint spans adds two model-approved
candidates, one on each of two previously abstained requests. In both
cases the cited windows come from the same profile field. None of the
23 proposed pairs has a top-three window for every criterion. The
rank-guarded output is identical to the baseline.

The two additional exact views change at most two of 400 fixed Top-5
slots. With every other view preserved, feasible changes in supported
P@5 range from zero to +0.50 percentage points, conditional on this
development trace. This is a bound over possible review outcomes.
It gives neither a labeled precision estimate nor a confidence
interval. Two independent reviewers still need to assess the exact
views for relevance, qualifications and attribution.

Each arm runs 184 recovery verifier pairs. With the disjoint-span
configuration, the two alternatives execute 31,562 recovery tokens
versus 35,546 for the baseline. Serial P95 is 1.061–1.080 s for the baseline and 1.081–1.095 s
for open combination. Timing includes proposal search and inference.
Index construction and service queueing are separate. Both replay
orders reproduce all non-timing outputs.

## Reproduce

Restore the restricted projection, role windows, source vectors and
pinned model checkpoints from the
[enterprise first-query experiment](../enterprise-role-first-query-v1)
and [packed recovery reference](../enterprise-packed-role-search-v1).
Run on a Slurm GPU allocation with the same checkpoint and manifest
hashes. Each execution uses new public summary and private trace paths.
Repeat with `--reverse` to change the initial arm order. The runner
also rotates order across requests. Private traces contain query IDs,
criterion offsets and exact citation offsets.

```sh
export PYTHONPATH=src:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.enterprise_guarded_evidence_bundle \
  QUERIES CORPUS ROLE_WINDOWS VECTORS VECTOR_META MODELS PACKED_REFERENCE \
  SUMMARY PRIVATE_TRACE --proposal-depth 256 --allow-disjoint-windows
```

Omit `--allow-disjoint-windows` for the distinct-field configurations
and set `--proposal-depth` to 24 or 256 as appropriate. [Aggregate results](results.json)
record the exact runner hashes.
