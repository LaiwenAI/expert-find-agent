# Refill only role-incompatible slots from the title index

The [fixed mixed-view replay](../role-field-ranking-v1) reserved two
of four reads for titles regardless of the available role evidence.
The [CoF transfer study](../cof-title-route-transfer-v1) found that
routing from score gaps did not consistently select cases that
benefit. Here the trigger is a structural hole in the original
four-source prefix: a complete available declaration rules out the
requested pair of roles for the same person in that source.

Keep every original source that can provide the pair or whose
declaration is incomplete or unavailable. Fill the remaining slots
from the title rank, screening only the same structurally excluded
sources. If every original source remains usable, skip the title
scan. If there are fewer than four eligible sources, fill with the
unused original sources. All four-read arms send four distinct
sources to the same topic verifier in one batch. The original
eligible read candidates are preserved; the five-slot final output
can still change when new evidence outranks an original citation.

Compare topic four, topic eight, four role-eligible sources from the
hybrid order, fixed role-aware mixing, and gap-triggered refill.
When there are enough eligible sources, role-eligible hybrid
selection is equivalent to keeping the surviving original prefix
and filling holes from the hybrid rank. This isolates the value of
the alternate index from the value of role screening.

Use the same PLOS and eLife catalogs, role pairs, declaration masks,
Qwen embedding and topic reranker as the prior mixed-view replay.
Recheck the frozen read, issued, tagged, and off-label control
counts. Record the number of title scans, exact original witnesses
lost, broad topical labels, and eLife major-topic evidence at the
read, pass and final-citation stages. Timing includes query encoding,
retrieval and local verification with a warm serial batch. Topic
labels describe the publication. They are proxies for evidence
relevance, and the replay does not establish industrial expertise.

## Results

Slurm jobs `44011` and `44012` on ws5 match all non-timing results.
The four-source controls reproduce the frozen replay. The table uses
the full available declaration mask. All source counts and citations
come from a fixed verifier and threshold.

| Panel and route | Title scans | Broad tagged / off-label citations | Major citations | Queries with a major citation | Original witness pairs lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| PLOS topic four | 0 / 46 | 37 / 4 | — | — | 0 |
| PLOS hybrid role refill | 0 / 46 | 52 / 16 | — | — | 0 |
| PLOS fixed role-aware mix | 46 / 46 | 46 / 11 | — | — | 6 |
| PLOS title refill | 42 / 46 | 48 / 11 | — | — | 0 |
| eLife topic four | 0 / 110 | 142 / 83 | 19 | 9 | 0 |
| eLife hybrid role refill | 0 / 110 | 154 / 101 | 18 | 10 | 9 |
| eLife fixed role-aware mix | 110 / 110 | 152 / 86 | 27 | 13 | 45 |
| eLife title refill | 76 / 110 | 151 / 94 | 24 | 12 | 6 |

In eLife, among the 31 requests with a known same-person major-topic
source, gap-triggered refill reaches 15 at selection, 14 after the
topic threshold, and 12 in the final citations. Hybrid role refill
reaches 12, 12, and 10. The paired difference in major-topic slots per
five moves by +1.09 percentage points, with a descriptive 95%
topic-cluster interval of 0 to +2.55 points. The fixed mix attains
16, 14, and 13. At partial declaration visibility the title scans
drop to 56 of 110, and 11 queries return a major citation.

On PLOS, the title refill produces four fewer tagged and five fewer
off-label citations than hybrid role refill at full availability.
Its tagged-slot difference is −1.74 percentage points with a
descriptive interval of −6.09 to +1.74. At partial visibility the
two policies tie at 41 tagged citations.

The new route retains major-topic coverage with fewer title scans
than fixed mixing, while remaining a trade-off against the already
strong hybrid role refill. Loss of an original witness is counted
by exact expert and source pair. Fewer title scans do not establish
an online speedup: for eLife, hybrid role refill, fixed mixing and
gap refill all have P95 near 29.4–29.5 ms in job `44011`.
The local catalogs contain 78 PLOS and 322 eLife sources.

[Aggregate results](results.json)

## Reproduce

Rebuild the pinned projections and use the inputs from
[the fixed mixed-view replay](../role-field-ranking-v1). Run the same
Slurm GPU command with an added mixed-view reference before the model
paths:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.role_gap_title_refill PLOS_PROJECTION \
  benchmarks/plos-topic-role-holdout-v1/manifest.json benchmarks/plos-bounded-topic-v1/results.json \
  ELIFE_PROJECTION benchmarks/elife-mesh-v1/mesh.json benchmarks/elife-mesh-scope-v1/results.json \
  benchmarks/role-field-ranking-v1/results.json QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
