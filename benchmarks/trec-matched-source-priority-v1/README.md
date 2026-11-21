# Compare static document priority against the learned scheduler

A [controlled TREC replay](../trec-source-owner-contrast-v1) found
that selecting sources by the sum of their top two estimated
supports confirmed four more slots than reciprocal expert-rank
selection, under exactly 147 reads and checks. The next comparator
is the [published learned scheduler](../trec-coverage-scheduler-v1).

Reuse the same 49 held topics, fixed 20-person shortlist,
depth-16 source menus, and leave-one-topic-out pair model as the
published scheduler. For each held topic, reproduce the original
greedy scheduler at the reuse policy's realized caps. Its own
realized cost bounds the new routes. The two static rules reserve
up to that number of source documents before querying any held
labels.

- **Rank sum** prioritizes a document by the sum of reciprocal
  expert ranks across its menu entries.
- **Top two** prioritizes a document by the sum of its two
  highest estimated expert–document support probabilities.

The exact same learned scheduler then checks sources within each
reserved set. A separate retrospective control reruns the original
scheduler against each static rule's realized read and check counts,
per topic. Each control can use fewer reads or checks if it exhausts
eligible actions. The original scheduler's internal order is
unchanged. TREC grade-2 support and relevance determine observed
credit. Unjudged expert–document pairs contribute no observed credit
and are counted separately. No text is read and no service is timed.

## Results

| Route | Reads | Checks | Unknown checks | Supported / issued |
| --- | ---: | ---: | ---: | ---: |
| Original learned scheduler | 80 | 266 | 220 | 10 / 11 |
| Rank sum | 76 | 227 | 200 | 1 / 2 |
| Scheduler at rank-sum work | 71 | 223 | 183 | 9 / 10 |
| Top two | 80 | 221 | 190 | 5 / 5 |
| Scheduler at top-two work | 77 | 221 | 185 | 8 / 9 |

Top two improves over rank sum. At the same realized number of
checks, the published scheduling policy confirms three more slots
with three fewer reads. With that per-topic accounting, top two
changes observed supported P@5 by −1.22 percentage points relative
to the original scheduler (paired 95% bootstrap interval −3.67 to
0.00). The interval and incomplete judgments limit inference beyond
these topics. Keep the published scheduler as the stronger observed
control. The gain over reciprocal-rank document selection does not
justify reserving the top-two shortlist globally.

Slurm tasks 46872_0 and 46872_1 ran on ws3. They reproduce the
published greedy reference and match summary counts and topic
signatures in opposite topic orders. The first three routes also
reproduce the cap-only pilot, array 46860. Both experiments reuse the
same held TREC topics. [Aggregate results](results.json).

## Reproduce

Restore the official TREC topics and qrels with the pinned digests
in the aggregate results and use a fresh path under Slurm.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.trec_enterprise.matched_source_priority TREC_DATA \
  benchmarks/trec-coverage-scheduler-v1/results.json NEW_RESULTS
```

Repeat with `--reverse` and a different output path.
