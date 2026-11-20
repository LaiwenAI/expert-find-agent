# Reuse verified role declarations across distinct requests

Use the PLOS and eLife fixtures, held-role folds, exact
top-five author order and minimum citation reference from
the [column index experiment](../role-column-index-v1).
Use the [four-source wave planner](../role-index-read-waves-v1)
in every arm. In each fold, warm-start the offline placement
from the same source-read traces of role combinations which
omit the held role. Replay all 78 distinct held combinations
once per order. The two deterministic shuffled orders each
start with an empty online cache.

Compare whole-source and smoothed-column offline placements
at the full 25% byte cap, a pure LRU-style cache at the same
cap, and both offline placements with half the cap reserved
for online reuse. Cache only complete, verified source
declarations observed during earlier requests. Refresh an
entry if a later response cites that source. Insert newly
fetched entries in fetch order after completing the current
request. The byte accounting uses the same packed whole-source
representation as the static index. If a cached declaration
also occupies partial columns in the static index, charge
both. This measures encoded payload, excluding Python objects.
Each replay checks exact rank and citation order on every query.

Report source reads, fetched declaration bytes, read waves,
and cold-start reads. This is a source-access replay over
immutable snapshots. It excludes HTTP overhead, invalidation,
traffic concurrency and changes in contributor declarations.
The shuffled stream explores a shift in role requests. It
does not estimate the effect under real user traffic.

## Results

Both Slurm CPU tasks in array 45313 completed on ws3.
Every policy exactly reproduces all 312 ordered results
per publisher and per request order. The static controls
reproduce the published source-read and read-wave totals.

| Policy | PLOS reads, 525 bytes | eLife reads, 1,464 bytes |
| --- | ---: | ---: |
| Whole-source static | 1,173 | 2,663 |
| Smoothed-column static | 1,500 | 2,438 |
| Online cache only | 1,814–1,841 | 3,188–3,241 |
| Half whole-source plus cache | 1,556–1,562 | 2,746–2,778 |
| Half smoothed-column plus cache | 2,681–2,709 | 2,887–2,890 |

Ranges are the two shuffled orders. Relative to the better
full static control for each publisher, the better tested
hybrid uses about 33% more reads on PLOS and 13–14% more
on eLife. The eLife whole-source hybrid avoids an origin
read on 74–81 of 312 requests, versus 13 with the best
static column control, while using more total reads and
waves. Coverage at zero origin reads and total source cost
therefore favor different policies. A fixed half-cap
reservation for this reuse policy is unsupported at this
workload and byte budget. These results leave other access
patterns, capacity allocations and admission rules open.

[PLOS aggregates](plos.json) · [eLife aggregates](elife.json)

## Reproduce

Prepare the same corpora and index result file as the
[column index experiment](../role-column-index-v1).
Run once per publisher on a Slurm CPU allocation with a
fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.role_index_cross_query_cache \
  CORPUS COLUMN_INDEX_RESULTS SCRATCH NEW_RESULTS --publisher plos
```

Pass `--publisher elife` for eLife.
