# Preserve citation decisions when compacting topical evidence checks

The [holdout experiment](../plos-topic-role-holdout-v1)
found changes in emitted witness lists after dropping articles
known to lack either requested same-person duty. Check the
scores of identical eligible articles before and after removing
irrelevant rows, with the model, token IDs and query fixed.

Use the same topic ranker, article candidates, declarations,
and topical acceptance cutoff as the holdout comparison.
Each request begins with the original eight source candidates.
Known nonmatching declarations may be skipped. All articles
with unresolved declarations stay eligible. Keep exactly the
same person binding and witness tie-breaking order.

| Arm | Model batch |
| --- | --- |
| Base | All eight sources in their original order |
| Compact | Only eligible sources, with dynamic padding |
| Fixed width | Eligible sources with the original full batch's padding width |
| Margin guard | Compact, then rerun Base on an ambiguous acceptance or order margin |
| Grouped | Consecutive original source pairs, each with its own padding |
| Grouped skip | The same source pairs, skipping only pairs without an eligible source |

The margin guard uses the maximum absolute difference on each
eligible source between Base and Compact in the original pilot
catalog, evaluated over both declaration masks. Freeze this
tolerance for the disjoint test catalog. Rerun Base if any
eligible score is within that distance of the acceptance cutoff,
or any two adjacent above-cutoff scores differ by at most twice
that distance. The observed calibration envelope is empirical.

The fixed-width comparison isolates the change in the maximum
padded length while still changing the number of rows.
Grouped skip preserves input geometry within each retained
group and uses Grouped as its exactness reference. Also
compare both grouped arms to the original eight-row route.

Use the same 46 requests and declaration masks as the previous
holdout. Report changes in relevant scores, threshold crossings,
ordered person-and-citation outputs, and padded input token slots.
Timing measures online topic retrieval, shared tokenization for
all eight original sources, model execution, and witness selection.
Task dispatch, network transport, upstream request parsing,
and service queues are outside this replay.

## Results

Jobs `43476` and `43479` reproduced all outcome and
matching-score counts on one RTX 5090. Calibration covers
236 pilot request-mask observations and 1,123 eligible
scores. The frozen margin is 0.375. Timings below come
from job `43479`.

| Indexed declarations | Arm | Ordered witness lists changed / 46 | Eligible score changes | Threshold flips | Padded slots | P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | Base | 0 | 0 | 0 | 200,496 | 64.46 ms |
| 42 | Compact | 1 | 74 | 0 | 131,128 | 56.19 ms |
| 42 | Fixed width | 0 | 47 | 0 | 136,111 | 56.85 ms |
| 42 | Margin guard | 0 | 43 | 0 | 251,272 | 105.33 ms |
| 78 | Base | 0 | 0 | 0 | 200,496 | 65.29 ms |
| 78 | Compact | 2 | 45 | 1 | 83,137 | 47.82 ms |
| 78 | Fixed width | 0 | 22 | 0 | 87,456 | 48.26 ms |
| 78 | Margin guard | 0 | 24 | 0 | 155,481 | 98.97 ms |

Fixed-width compaction reproduces all observed ordered people
and witness citations, with 49 grounded and 14 off-label
citations under both masks. The paired P95 interval versus
Base is −12.12 to −1.42 ms with 42 indexed declarations
and −18.66 to −2.81 ms with 78. Retaining width changes
which numerical deviations affect the final answer. Many
raw margins still change, so exactness is an observed
property of this replay. The margin guard reruns the full
batch for 28 and 17 requests, respectively, increasing P95.

Grouped skip reproduces all eligible scores and witness
lists relative to its own grouped reference. The grouped
reference changes one witness list per mask relative to
Base. Grouped skip has P95 of 60.94 and 59.77 ms and
inspects 318 and 222 sources, respectively. These results
support testing fixed-width compaction on independent source
collections and on enterprise evidence with explicit
declaration coverage. [Aggregate results](results.json)

## Reproduce

Regenerate the two projections from the
[original pilot](../plos-topic-role-postings-v1) and
[disjoint holdout](../plos-topic-role-holdout-v1) snapshots.
The frozen holdout manifest verifies both projections.
Run on an allocated CUDA node.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.stable_topic_compaction \
  PILOT_PROJECTION HOLDOUT_PROJECTION HOLDOUT_MANIFEST QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
