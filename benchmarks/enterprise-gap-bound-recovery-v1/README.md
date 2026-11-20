# Reuse prior support in first-query role recovery

Take the same 80 authorized development requests, source index,
and frozen parser and verifier as the
[live qualification search](../enterprise-shared-qualification-v1).
Run the same automatic primary search once per request and keep
every primary recommendation fixed. The rescue trigger remains an
empty recommended list. A partially supported candidate can
retain its existing source citation for an individual criterion.

| Policy | Evidence selection and checks |
| --- | --- |
| Global full | Keep the two existing global role windows and score all criteria on each |
| Reuse | Keep those windows and score only the missing criteria for the same candidate |
| Gap reserve | Keep the first global window. If two windows were available, offer the second slot to an initially partial candidate with a window covering missing criterion terms |

The one-batch check uses the same 0.6B verifier and threshold.
A positive comparison for every missing criterion is required
before proposing the owner. Earlier citations carry their
original immutable offsets. A simple English and Chinese
common-object cue triggers single-source reuse: if any prior
citation comes from another source, the new window must support
every criterion by itself. Without a cue, same-person citations
may come from different sources. The cue list only captures
an explicit subset of scope dependence, and a paired evidence
bundle still needs independent review.

All arms use identical window counts per request. They can only
match or reduce the global baseline's extra criterion-pair count.
Timing combines a shared measured primary path with each arm's
rotated, measured rescue. It covers retrieval and verification
in the local serial replay. Timing excludes queues and external
evidence lookup. Keep source provenance and request-level outputs
in the authorized private workspace.

## Results

Jobs `43488` and `43491` reproduced the rescue outcomes on
one RTX 5090. Both replay the original candidate and query
inventory. Four requests contain an explicit common-object
cue. The three arms inspect the same 92 role windows.

| Policy | Additional criterion checks | Reused checks | Requests with provisional output | Rescue P95 | Shared primary plus rescue P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global full | 184 | 0 | 37 | 28.6 ms | 991.9 ms |
| Reuse | 182 | 2 | 37 | 29.5 ms | 991.8 ms |
| Gap reserve | 179 | 5 | 37 | 29.0 ms | 992.5 ms |

Gap reserve changes the second window in three requests.
All 73 provisional recommendations and their citations agree
across arms, including three recommendations from the rescue
stage. No policy produces a multi-source promotion.

A separate exhaustive diagnostic examines the 14 eligible
requests that abstain after primary verification and contain
one or more partially supported candidates. Among 32 partial
candidate slots, 12 have a scope-compatible role window.
The verifier scores all 58 such windows in batches outside
the timed path, performing 73 missing-criterion checks.
It completes zero candidate bundles. Five further windows
are excluded by the explicit common-object guard. This
diagnostic covers the indexed role windows of the same
primary candidates. It leaves open deeper general sources,
additional candidates, improved verification, and human
judgments of the existing evidence.

Keep the global full check. The observed budget savings do
not translate into a material latency reduction. A different
schedule on this same-owner role-window pool would need new
support evidence before adding coverage. [Aggregate results](results.json)

## Reproduce

Use the same pinned development artifacts and checkpoint layout
as the [live role-index study](../enterprise-role-first-query-v1).
Run the GPU benchmark under Slurm. The public report contains
aggregate counts and timing.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.enterprise_gap_bound_recovery \
  DEVELOPMENT_QUERIES DEVELOPMENT_SOURCES ROLE_WINDOWS VECTORS_NPY VECTOR_METADATA \
  MODEL_ROOT NEW_AGGREGATES PRIVATE_TRACE
```
