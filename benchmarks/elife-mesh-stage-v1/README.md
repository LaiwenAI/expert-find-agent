# Locate the loss of reachable major-topic evidence

Instrument the [frozen 110-query eLife replay](../elife-mesh-scope-v1).
Do not change the topic ranker, read list, verifier, ranking or
recommendation cutoff. Validate all non-timing metrics against both
published runs before emitting the diagnostic.

An available target is a source in the full corpus with the major
[MEDLINE descriptor](https://www.nlm.nih.gov/mesh/meshhome.html) and the
two requested roles explicitly assigned to the same author. At each
stage, count requests with at least one such target. This is a
query-level measure, independent of how many slots cite that source.
The successive stages are:

1. At least one target is among the articles actually read.
2. At least one of those targets passes the frozen topical verifier.
3. At least one passing target is explicitly cited for a returned expert.

A loss after the verifier is further classified. If every possible
target owner is in the result with a different citation, it is a
citation replacement. If at least one such owner is absent and all
five slots are occupied, it is an output-budget omission. For each
stage, compare changed queries as well as aggregate counts. A zero
net change can conceal both added and lost sources.

This diagnoses an indexed major-topic signal under the exact
structured role benchmark. Sources without an assignment are not
declared irrelevant, and the reachable set is not an estimate of all
qualified experts. The instrumented full replay keeps ordinary
descriptor metrics as the equivalence checks. Stage numbers are
conditional on the 31 queries with a major same-person target.

## Results

RTX 5090 job `43812` reproduced the published non-timing replay
results. For the 31 reachable queries, the first eligible major
source appears among topic ranks 1–4 for 11, ranks 5–8 for four,
and beyond rank 8 for 16.

| Policy, 280 declarations indexed | Selected | Passes verifier | Issued with exact major witness | Missing from reads / verifier loss / output loss |
| --- | ---: | ---: | ---: | ---: |
| Topic four | 11 | 11 | 9 | 20 / 0 / 2 |
| Topic eight | 15 | 12 | 9 | 16 / 3 / 3 |
| Person four | 12 | 12 | 10 | 19 / 0 / 2 |
| Top-eight person four | 11 | 11 | 9 | 20 / 0 / 2 |
| PLOS-calibrated topical floor | 13 | 13 | 10 | 18 / 0 / 3 |

All rows count queries. Topic eight adds a final major-witness
query and loses a different one relative to topic four. The
unrestricted role route adds one and loses none. The topical floor
adds two major-source selections and turns one into an issued
major-witness query. Every post-verifier loss in the table has
at least one possible owner omitted at a full five-slot budget.

The larger observed bottleneck is putting a relevant source in
the four-source pool. A perfect verifier or a different final
ranking restricted to the same four-source pool cannot address
the 18–20 queries where the four-source methods read no major
target. That observation localizes the improvement opportunity
on this dataset. It does not justify replacing the verifier or
using MeSH labels as model features in future validation.

[Aggregate stage counts](results.json)

## Reproduce

Use the [previous replay command](../elife-mesh-scope-v1) with the
following additional arguments in a Slurm CUDA job:

```sh
--diagnostic-reference benchmarks/elife-mesh-scope-v1/results.json \
--diagnostic-output NEW_DIAGNOSTIC_JSON
```
