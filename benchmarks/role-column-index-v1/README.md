# Index verified role columns instead of complete author declarations

Reuse the [PLOS](../plos-budgeted-declaration-index-v1) and
[eLife](../elife-budgeted-declaration-index-v1) structured role
benchmarks. In four folds, allocate index capacity with the
286 offline queries that omit one role, then test 78 queries
requiring it. Keep the candidate prior and exact top-five
minimum-citation requirement. The eLife panel contains only
the 280 articles with complete, parseable declarations.

Within a paper, store one packed author bitset per indexed
role. A four-byte header gives the source ordinal and known
role mask. Absent role columns remain unknown unless the
article ceiling already excludes them. In each query the
planner uses lower and upper masks. It returns the optimistic
minimum bundle only when verified evidence in that same bundle
covers the required roles. Otherwise it loads a relevant
publisher declaration and tightens those bounds. All policies
use the same planner and the same binary encoding, so the
comparison isolates the allocation of a fixed payload.
The common corpus metadata and Python resident object overhead
are outside the byte cap. The cap counts the actual packed bytes.

## Placement controls

- **Whole author / whole cold:** allocate all columns for
  any selected article. Use either earliest visible author order
  or offline opens per encoded article byte.
- **Column frequency:** use offline opens containing each
  role per column byte.
- **Column bundles:** greedily purchase the remaining role
  columns needed to complete the most frequent source-level
  query masks per byte. Opening one paper can satisfy several
  such masks.
- **Smoothed columns:** add one-third of each source's total
  opens to every applicable role's frequency before ranking.
  This positive prior uses all article roles independently of
  which one is held. It was added after inspecting the unsmoothed
  results on these same panels and remains exploratory.

## Results

Both Slurm CPU tasks per publisher in array `45046` on `ws3`
reproduce all byte and file counts. Every query and timing
repeat matches the authoritative ordered experts and exact
shortest citations. Complete packed payloads are 2,100 bytes
for PLOS and 5,857 bytes for eLife. The following fixed quarter
caps are 525 bytes and 1,464 bytes, respectively.

| Publisher, 25% cap | Whole author | Whole cold | Column frequency | Smoothed columns | Column bundles |
| --- | ---: | ---: | ---: | ---: | ---: |
| PLOS file opens | 1,744 | 1,173 | 2,953 | 1,500 | 4,379 |
| eLife file opens | 2,991 | 2,663 | 3,188 | 2,438 | 6,175 |

Counts sum the four held folds. Both unsmoothed column variants
index zero instances of the excluded role at the tested byte
caps. Smoothing indexes 79 and 125 held-role columns at the
quarter cap across PLOS and eLife folds. The wholly held role
is a deliberate stress case beyond ordinary frequency shifts.

At eLife's quarter cap, smoothed columns use about 8% fewer
opens than whole cold placement. The local planner and decoding
P95 drops by about 10–26% across the four paired folds.
At the same cap PLOS favors whole cold placement. At eLife's
half cap, whole author needs 214 reads, versus 298 for smoothed
columns. Unsmoothed frequency can reduce local P95 even when
the aggregate file counts rise. The report therefore includes
decoded bytes per query as well as file counts.

The exactness property applies to verified role columns and
valid article ceilings. The allocation results are conditional
on the publisher, budget and role split. The warm component
measurement excludes encoding and decoding the resident index,
query parsing, candidate retrieval, and remote source reads.
[Aggregate payload, read and timing results](results.json).

## Reproduce

Use the pinned PLOS fixture or the eLife projection and run
on a Slurm CPU allocation with a fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.role_column_index \
  CORPUS benchmarks/elife-role-nomination-v1/manifest.json SCRATCH NEW_RESULTS \
  --publisher elife
```

For the PLOS fixture, pass `--publisher plos`.
