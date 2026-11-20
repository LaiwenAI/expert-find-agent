# Batch source reads for exact expert proofs

Keep the publisher projections, fixed author order, held role
requests, and 25% encoded-byte cap from the
[role-column experiment](../role-column-index-v1). Within the
first five still-possible authors, use resident positive role
evidence and article bounds to compute the optimistic canonical
citation bundle. Rank the unresolved sources by how many active
bundles use them, then the number of unresolved roles and source
identifier. Read at most one, two, or four per wave. Advance
past an author once its optimistic bundle becomes impossible.
A returned bundle has verified support for every requested role.

The serial control runs the existing owner-by-owner planner.
The width-one frontier control separates source scheduling from
concurrent reads. The test materializes source responses and counts
the batches that could be fetched concurrently. It does not
exercise network transport. Each result is compared with the
original exact author ordering and shortest citation bundle.
Check the per-request source set, total source bytes and tail
read rounds under both tested index placements.

## Results

Both completed tasks of Slurm array `45143` reproduce the
reference ranking and canonical citations for all 312 held
requests per publisher. Every policy reads the same source set
as the original serial planner for each individual request.
The throughput opportunity follows from grouping source reads.

| Publisher | Placement | Source reads | Serial rounds | At most two per round | At most four per round | P95 rounds, serial to four |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PLOS | Whole cold | 1,173 | 1,173 | 846 | 780 | 11 → 8 |
| PLOS | Smoothed columns | 1,500 | 1,500 | 957 | 873 | 12 → 8 |
| eLife | Whole cold | 2,663 | 2,663 | 1,767 | 1,515 | 31.45 → 19 |
| eLife | Smoothed columns | 2,438 | 2,438 | 1,536 | 1,319 | 25.45 → 14 |

With the better prior placement on each publisher, width four
reduces modeled source rounds by 33.5% on PLOS and 45.9% on
eLife. The per-request fetch-set identity is empirical and may
change with a new corpus or candidate order. The next transport
comparison must measure CPU planning, request overlap and
contention, since a read round is not a latency measurement.
[Aggregates](results.json).

## Reproduce

Use a Slurm CPU allocation with the pinned public projections
and the previous result manifest.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.role_index_read_waves \
  CORPUS benchmarks/elife-role-nomination-v1/manifest.json \
  benchmarks/role-column-index-v1/results.json SCRATCH NEW_RESULTS \
  --publisher elife
```

For the PLOS fixture, pass `--publisher plos`.
