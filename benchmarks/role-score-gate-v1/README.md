# Test a calibrated topical floor for role-based candidate replacement

An [empty-prefix rule](../role-empty-prefix-v1) avoided losing already
issued results but did not reproduce its tagged gain across publishers.
Here a cheap topical signal evaluates what could be added, before the
reranker reads articles. Reuse the query's original dense similarities
from hybrid retrieval. The starting reference is the lowest dense
similarity among the first four hybrid candidates. Preserve all
original candidates with a same-person match or an incomplete
declaration. A new candidate enters only if its source is role
eligible and its dense similarity is within a fixed margin of that
reference. Work through the existing topic rank and refill any unused
budget with the highest-ranked original sources. Verify exactly four
sources once. A missing declaration is eligible throughout and never
justifies removing an original candidate.

Select one margin on the [original PLOS pilot](../plos-topic-role-postings-v1)
only. An initial grid used a no-replacement control and
`0, 0.01, 0.025, 0.05, 0.10`. No nonnegative margin passed.
The follow-up also tests `−0.10, −0.05, −0.025, −0.01, −0.005`, requiring
that new sources score higher than the weakest of the original four.
Require no more than a one-point increase
in off-label citations per five output slots against topic four under
either declaration mask. Among feasible margins maximize total
tagged recommendations, then prefer fewer off-label citations and
fewer injected sources. The rule uses publisher tags for development
and the same frozen reranker cutoff as previous experiments. The
off-label condition is a proxy and not a bound on human-judged risk.

Freeze the margin before replaying both masks in the
[PLOS holdout](../plos-bounded-topic-v1) and
[eLife](../elife-role-nomination-v1). Use topic four, topic eight,
unrestricted same-person nomination and same-person nomination
limited to the top eight as controls. Require these four controls
to reproduce previous counts. During the timed test every arm uses
live verification; only calibration reuses scores for identical
ordered batches. Times include online query encoding, rank, gating,
local verification and citation selection. They omit queues, index
construction and service overhead.

## Results

RTX 5090 job `43786` selected no replacement with the initial grid.
Job `43789` reproduced those calibration counts and selected a margin
of −0.01 from the expanded grid. This means an injected source had to
score at least 0.01 higher than the weakest original source in dense
similarity. On the pilot, with all available declarations it raises
tagged recommendations from 87 to 102 and off-label citations from 35
to 40. The proxy ceiling holds for this setting on both pilot masks.

| Test panel and declarations | Topic four | Similarity floor | Unrestricted person four |
| --- | ---: | ---: | ---: |
| PLOS, 42 indexed | 37 / 4 | 43 / 4 | 41 / 7 |
| PLOS, 78 indexed | 37 / 4 | 44 / 4 | 52 / 16 |
| eLife headings, 280 indexed | 15 / 7 | 15 / 8 | 15 / 7 |
| eLife keywords, 144 indexed | 64 / 30 | 68 / 32 | 69 / 31 |
| eLife keywords, 280 indexed | 64 / 30 | 74 / 40 | 80 / 46 |

Cells are tagged / off-label recommendation slots. At full available
coverage, the floor's tagged P@5 gain over topic four is 3.04 points in
PLOS, with a descriptive paired 95% interval of +0.43 to +6.09. In eLife
keywords the gain is 4.17 points, with interval +0.83 to +9.17. The eLife
keyword proxy risk cap is exceeded, with ten additional off-label slots
out of 240 potential positions. The gated route still fails the
preexisting noninferiority checks versus topic eight in both panels.

All methods in the table verify four sources per request. The floor
requires no additional model call. Full-coverage local P95 for topic
four and the gated route is 33.38 and 35.66 ms in PLOS, and 27.19 and
27.39 ms on eLife keywords. Selecting later sources changes prompt
length and model execution even at fixed source counts. The added
off-label slots remain unjudged. The experiment supports treating the
calibrated floor as a comparator and evaluating topical verification
separately from role eligibility on genuinely new domains.

[Aggregate results for both grids](results.json)

## Reproduce

Use the frozen projections and verified checkpoints from the original
experiments in one Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.role_score_gate PLOS_PILOT PLOS_HOLDOUT \
  benchmarks/plos-topic-role-holdout-v1/manifest.json benchmarks/plos-bounded-topic-v1/results.json \
  ELIFE_PROJECTION benchmarks/elife-role-nomination-v1/manifest.json \
  benchmarks/elife-role-nomination-v1/results.json QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
