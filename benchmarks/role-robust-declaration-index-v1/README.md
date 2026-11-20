# Reserve author-prior index space under a shift in requested roles

The [eLife transfer result](../elife-budgeted-declaration-index-v1)
found that optimizing average development reads picked cold-read
placement even when the author-prior placement had fewer reads
on held role combinations. Use the same publisher snapshots,
capacity limits, candidate order, role requests, and exact proof
routine in [PLOS](../plos-budgeted-declaration-index-v1) and eLife.
Expand the role holdout from four positions to all 14. A fold
fits on 286 triples missing a role and tests 78 triples requiring
it. Every triple appears in three test folds, so sums describe
the evaluation workload rather than independent observations.
The publishers have independent source and author projections.

## Mixed placement and calibration

Reserve 0%, 25%, 50%, 75% or 100% of the byte cap for the
author-prior placement. Use the remaining space for the local
cold-read order trained on the fold's development queries.
Every allocation obeys the same byte cap. The two endpoints
are the pure policies from the prior studies.

Freeze one mix per fold and byte cap using only development
read counts. Compare three selection objectives:

- **Mean:** choose the lowest total number of development reads.
- **Role guard:** among the 13 visible roles, find each mix's
  largest mean excess reads over author-prior, taking queries
  containing that role. Minimize that maximum.
- **Query-tail guard:** minimize the 95th percentile of per-query
  excess reads over author-prior.

Both guards break ties by mean reads, then by the fixed
mix order. Role strata overlap. Both scores are empirical,
so neither guarantees noninferiority on unobserved roles.

## Results

Both CPU tasks for each publisher in Slurm array `45025` on
`ws3` reproduce the held read counts. Every mix returns the exact
ranked expert identities and canonical minimum citations from
exhaustive enumeration. The previously reported four folds
also match both pure placement endpoints at every byte cap.

| Publisher, capacity | Author-prior reads | Mean-selected reads | Role-guard reads | Query-tail-guard reads |
| --- | ---: | ---: | ---: | ---: |
| PLOS, 25% | 10,563 | 4,450 | 5,249 | 5,592 |
| PLOS, 50% | 1,506 | 348 | 420 | 348 |
| eLife, 25% | 16,206 | 11,573 | 12,277 | 16,206 |
| eLife, 50% | 1,635 | 1,698 | 1,649 | 1,597 |

At eLife's 50% cap, mean selection opens more files than the
author-prior policy in seven of 14 held-role folds. Both guards
reduce this to one fold and reduce the largest loss from 51
extra file opens to 14. The role guard retains about 24% read
savings at the 25% cap. At that cap, the query-tail guard chooses
author-prior in every eLife fold and gives up the read savings.
On PLOS, the guards spend more reads than mean selection at the
25% cap, while still beating author-prior in every fold.

The role guard offers a less conservative trade-off on the
tested eLife capacities. Its one remaining regression and the
overlapping folds constrain the result. The study checks
structured author declarations in fixed catalogs. It does not
measure end-to-end search latency or cross-domain reliability.
[Fold-level reads, byte costs and local timings](results.json).

## Reproduce

Use the pinned PLOS fixture or the prepared eLife projection.
Run on a Slurm CPU allocation with a fresh result file.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.role_robust_declaration_index \
  CORPUS benchmarks/elife-role-nomination-v1/manifest.json SCRATCH NEW_RESULTS \
  --publisher elife
```

For the PLOS fixture, pass `--publisher plos`.
