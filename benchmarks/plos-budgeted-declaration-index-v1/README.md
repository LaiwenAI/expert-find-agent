# Allocate verified author declarations under a byte budget

The [shared-source study](../plos-shared-bundle-frontier-v1)
found that near-minimal source reads are unavoidable when only
article-level role ceilings are resident. Test which verified
author declarations should be indexed in advance. Use the same
152 PLOS articles, 580 author identities and 364 structured
three-role requests. Each policy certifies the ordered top five
and the canonical minimum citations with the same search code.
An uncached author row remains unknown until source read.

Charge each source the size of its compact author-to-role-mask
JSON payload. Set a budget of 10%, 25% or 50% of the 18,721
payload bytes. Place whole sources in priority order, skipping
one when it would exceed the budget. The byte cap excludes
in-memory object overhead, article ceilings and the rest of
the search index.

- **Author-prior:** index articles by the earliest author they
  can affect in the existing byline-based prior.
- **Optimistic proof per byte:** count appearances in the
  first five optimistic proofs in offline reference requests.
  This allocation sees article ceilings and author order.
- **Cold-read frequency per byte:** count how often each
  source actually needs to be opened in unindexed offline
  requests, divided by its payload size. This uses verified
  declaration values in the offline exercises and requires
  no prior user traffic.
- **Hash:** six independent fixed DOI-hash orderings.

Workload-specific priorities fit on the 286 role conjunctions
that omit one designated role. Evaluate on the 78 conjunctions
that include it. Repeat with four fixed role positions. Folds
share publishers, authors and some test combinations. The four
folds describe transfer across conditions in this fixed catalog.

## Fixed and changed author ranking

In the original ranking, favor the byline prior. For a separate
stress test, rank authors by the number of their papers whose
article ceiling overlaps the requested roles, then by the
largest overlap in a single paper, then by the original prior.
This uses metadata and does not read author declarations. It
changes the exact ranked output in 41–71 of 78 requests per
fold. Evaluate the existing byte allocations unchanged and
also refit the cold-read policy under the changed ranking,
using only that fold's offline training conjunctions.

## Results

Both tasks of Slurm CPU array `45005` on `ws3` reproduce all
bytes, file counts and exhaustive ordered-output assertions.
The table uses the 4,680-byte cap. Each row represents 78
requests. The hash column is the median across six inventories.

| Held role position | Fixed prior, hash | Fixed prior, author-prior | Fixed prior, cold-read | Changed ranking, author-prior | Changed ranking, unchanged cold-read index | Changed ranking, refitted index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,355.5 | 547 | 243 | 924 | 475 | 195 |
| 4 | 1,586.0 | 1,004 | 356 | 1,096 | 440 | 368 |
| 8 | 1,396.0 | 717 | 290 | 1,192 | 577 | 395 |
| 12 | 1,282.0 | 463 | 226 | 897 | 450 | 162 |

Cold-read placement reduces file opens by 51–65% relative to
author-prior placement at the same byte cap, with the same
ordered recommendations and citations. Warm planning plus
decoding P95 falls by about 32–43% in the fixed-prior comparison.
The precomputed index also reduces reads by 49–60% under the
changed ranking. Refitting improves file counts in all four
changed-ranking folds, with heterogeneous P95 effects. The
cold-read policy also outperforms every hash inventory in file
counts at all three tested byte caps and both rankings.

The timer includes local exact proof planning and warm decoding.
It excludes index construction, population of the cache, ranking
computation, model inference, network and raw XML fetching. The
result supports a workload-based index allocation for structured
publisher declarations within one catalog. Independent publisher
and free-text enterprise transfer still require evaluation.
[Aggregates and all byte caps](results.json).

## Reproduce

Run on a Slurm CPU compute allocation with a fresh result.
The source fixture and the ranking routine are hash-pinned.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.budgeted_declaration_index \
  benchmarks/plos-credit-role-postings-v1/sources.json SCRATCH_DIRECTORY NEW_RESULTS
```
