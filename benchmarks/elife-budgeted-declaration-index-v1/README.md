# Transfer a budgeted declaration index to eLife

Use the [pinned independent eLife snapshot](../elife-role-nomination-v1),
whose source selection excluded original PLOS titles and author
identifiers or exact names. Keep the 280 of 322 eligible eLife
articles whose publisher role declarations are complete and
parseable. This yields 2,292 author identities, with validated
cross-paper links or article-local author ordinals. The controlled
task certifies publisher-declared roles within that complete
subset. It does not use unparsed declarations as negatives.

Reuse the [PLOS allocation benchmark](../plos-budgeted-declaration-index-v1):
364 triples of declared roles, 286 development conjunctions and
78 held conjunctions per fold, three byte budgets, fixed author
prior, exact top-five identities and minimum citations. Repeat
with the same four held role positions. The folds share
authors, publications and some conjunctions.

## Transfer arms

- **Author-prior:** use the existing byline-order allocation.
- **Local cold-read allocation:** fit open frequency per byte
  using eLife development queries. This transfers the procedure
  from PLOS while rebuilding the priority list on eLife.
- **PLOS-only scoring model:** fit an L2-regularized linear
  model to the source-level cold-read rates in PLOS development
  queries. Inputs are normalized earliest and mean author rank,
  number of authors, linked-author share, article role-ceiling
  size, optimistic nomination frequency and payload size.
  Apply the frozen coefficients and scaling to eLife. Rank by
  predicted open frequency per byte. No eLife read outcomes
  enter this model. Metadata and payload sizes are available
  in the target index.
- **Optimistic and hash controls:** reuse the original
  metadata-only and six fixed hash orders.

The follow-up selector compares the first two arms for each
capacity using actual development read counts before held-query
evaluation. The follow-up reuses the same held panel after the
transfer diagnostic. The selection itself uses only the 286
development conjunctions.

## Results

Both Slurm CPU tasks in array `45021` on `ws3` reproduce
all storage and read counts, and pass the same output checks.
The exact author and citation results are checked against
exhaustive enumeration. Recorded bytes are compact per-source
JSON payloads before in-memory index overhead. Each cell reports
file opens across the 78 held conjunctions.

| Held role | Author-prior, 25% | Local cold-read, 25% | PLOS-only, 25% | Author-prior, 50% | Local cold-read, 50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 489 | 448 | 826 | 6 | 31 |
| 4 | 433 | 412 | 713 | 7 | 32 |
| 8 | 1,342 | 965 | 1,247 | 148 | 177 |
| 12 | 892 | 594 | 979 | 45 | 66 |

At the 17,771-byte cap (25%), local cold-read placement saves
5–33% of reads against author-prior placement. At 10%, it saves
12–21%. At the 35,543-byte cap (50%), the author-prior
allocation needs fewer reads in all four held folds. The
development read-count selector chose cold-read placement
for every fold and budget, including all four reversals at
50%. A decision calibrated on combinations omitting one role
can therefore fail on combinations requiring that role.

The PLOS-only model beats the author-prior allocation in all
four 10% folds, in one of the four 25% folds, and in none at
50%. This supports local calibration of the index allocation
and reveals a publisher and capacity boundary to transferring
a learned priority. The effect has not been shown on an
independent enterprise dataset.

Timing covers the warm local proof and decoding component.
It excludes index construction, source fetch, encoding,
candidate ranking and network service.
[Aggregate results](results.json).

## Reproduce

Rebuild the eLife projection with the XML preparation command
in the linked eLife benchmark. Run on a Slurm CPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.declaration_index_publisher_transfer \
  benchmarks/plos-credit-role-postings-v1/sources.json ELIFE_PROJECTION \
  benchmarks/elife-role-nomination-v1/manifest.json SCRATCH_DIRECTORY NEW_RESULTS
```
