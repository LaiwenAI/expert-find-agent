# Verify enterprise criteria with contiguous source windows

Transfer the public [claim-specific evidence comparison](../redocred-claim-bound-windows-v1)
to the live enterprise development route. Reuse the pinned 80
requests, 52,667 sources, parser, exact dense index, original top
five experts, and two source records per expert. The same guarded
packed reranker processes the same number and order of primary
criterion–source pairs across all arms.

| Arm | Source text shown for each criterion |
| --- | --- |
| Reference | The first 800 characters, as in the earlier guarded route |
| Short lead | The first 320 characters of the same source |
| Criterion window | The 320-character contiguous span within the original opening with most matching criterion terms |

Criterion-window candidates start at the beginning, sentence
boundaries, a 160-character grid, and the final aligned window.
Chinese character bigrams and Latin tokens share the existing
tokenizer. Break ties at the earliest offset. All selected spans
refer to original source offsets. Candidate and source
nominations are fixed before shortening. The abstention-triggered
role-index fallback is unchanged. It can receive a different
set of issued experts or be called on a different request.

Run each arm live on every request and rotate the order. Measure
primary verification and the full serial route. Compare issued
expert identities and ordering, primary source IDs, verified
spans, and fallback outputs. A source ID match is weaker than
a span match. Replacing a span can remove the qualifying context
even when the source and expert remain the same.

These counts describe agreement with the existing model pipeline.
No candidate or citation disagreement is labeled correct or
incorrect without independent review. The sealed requests are
unused, and the release contains aggregate data only.

## Results

Slurm job 43368 completed on an RTX 5090 on ws5. Each
arm checks 1,280 primary pairs and 184 fallback pairs. All
three routes issue a recommendation on the same 37 requests,
and fallback invocation and output agree on all 80.

| Arm | Primary executed tokens | Primary P95 | Full serial P95 | Requests with changed expert lists | Requests with changed primary sources / spans |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 385,612 | 114.5 ms | 1,069.0 ms | 0 | 0 / 0 |
| Short lead | 291,104 | 95.2 ms | 1,052.0 ms | 7 | 13 / 18 |
| Criterion window | 291,434 | 98.1 ms | 1,057.0 ms | 7 | 14 / 18 |

The criterion route lowers primary executed tokens by 24.4%
and primary P95 by 14.3%. Its full serial P95 is about 12 ms
lower in this replay. Each short route gains four individual
expert slots and loses four relative to the reference. Equal
issued-query counts conceal changes to specific recommendations.
The criterion selector does not clearly improve on the short
lead under the same span budget. Direction and uncertainty
of quality differences remain unresolved until blinded
review, so retain full openings for issued claims while
testing shorter windows as proposals. [Aggregate results](results.json)

## Blinded evidence review

The replay can write a restricted trace containing response IDs,
expert IDs, and original citation offsets for the reference and
criterion-window routes. Its manifest binds the trace to the
aggregate execution and the pinned query and source inventories.
The packet builder takes every request with different issued
identities, order or citation views, plus up to four unchanged
issued requests selected by fixed hash.

Deduplicate on the exact request, person and citation span set.
Randomize anonymous case order and the source aliases shown
within each case. Preserve complete requirement text, given
expert identity, and exact displayed excerpts for attribution.
Only the private coordinator map records the methods and ranks.
Reviewers may still infer some assignments from excerpt length.
The packet is a diagnostic disagreement set and does not
estimate population prevalence or closed-set ranking precision.
The sealed cohort remains unused.

Slurm job 43375 reproduces the preceding recommendation and
source agreement counts. The packet covers 18 changed requests,
including seven with different experts or order and 11 with
evidence-only changes, plus four exact-view controls. The two
policies issue 51 slots each in this review subset. Deduplication
yields 80 review cases, with 22 exact shared views and 25
request–expert pairs that have different citation views. Both
reviewer copies pass evidence-binding validation. The blank
reconciliation marks all 80 cases missing on each of the three
fields. No review has been distributed or collected.
[Packet aggregate](review-packet-aggregate.json)

Use the two copies independently. The existing checker validates
unchanged evidence and requires a reviewer ID and rationale for
any labeled field. The existing reconciliation tool preserves
disagreements and unresolved fields for adjudication. Keep all
private files local and share aggregate counts only.

## Reproduce

Use the authorized [enterprise development resources](../enterprise-role-first-query-v1)
and pinned models on a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_criterion_windows \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS ROLE_WINDOWS VECTORS.npy \
  VECTOR_METADATA MODELS NEW_AGGREGATE_RESULTS
```


To prepare the restricted trace, include `--review-trace PRIVATE_TRACE_JSON`
in the preceding replay. Build the review packet locally against the
private full metadata catalog from the same development dataset.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.build_enterprise_window_review \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS PRIVATE_DATASET \
  PRIVATE_TRACE_JSON NEW_AGGREGATE_RESULTS PRIVATE_PACKET_DIR

PYTHONPATH=src:. python -m experiments.retrieval.reconcile_joint_review \
  PRIVATE_PACKET_DIR REVIEWER_A_JSONL REVIEWER_B_JSONL --output PRIVATE_RECONCILIATION
```
