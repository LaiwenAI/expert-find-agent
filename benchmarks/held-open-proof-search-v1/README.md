# Search and verify linked contributor evidence over the full eLife catalog

Extend the [held proof-group comparison](../engineering-proof-group-calibration-v1)
from known source pairs to catalog search. Merge the two pinned
eLife projections while preserving publication and author linkage.
Each article's title and first abstract portion enter the index.
The last portion supplies an excerpt query, or up to eight of its
rarest indexed words for a keyword query. The builder splits on
sentence boundaries when available and on word boundaries otherwise.
The benchmark provides the order and boundaries of the two
source and contribution conditions.

The frozen panels supply article- and recorded-author-disjoint fit
and held groups. For each group test both sources present, the
left withheld, and the right withheld. The designated source is
removed from both sides of rank generation when withheld. Use
the same BM25 and dense reciprocal-rank fusion and depth-six
candidate list for all methods. Each planned pair must contain
distinct articles sharing a resolved author.

| Source route | Ranking and acceptance |
| --- | --- |
| Linked | Smallest sum of hybrid ranks among linked pairs |
| Anchored | Linked rule restricted to at least one hybrid leader and no cosine drop above 0.04 on either side |
| Individual | Fit source membership on held-aside development groups and rank each pair by its weaker source score |
| Paired | Fit a direct pair classifier over the min and max of the five source signals and cross-query cosine difference |

The individual and paired models fit on 18 groups from the
development projection. Use the other 12 development groups to
set a strict cutoff at the worst substitute score in each clue
view. No held outputs enter this fitting. Their source calibration
group count is small. The cross-query cosine feature measures
whether a candidate document is a better match for its designated
clue than for the other clue.

After proposing sources, score the frozen ridge role bank. A valid
proof names a linked contributor and quotes the highest-scoring
declared role in each selected source. Count author correctness and
strict proof correctness separately. A strict proof requires the
exact designated pair, a gold author and the intended role on both
sides. Alternative papers count as unsupported under this exact
study contract. The source comparison uses the archived role cutoff.

Independently compare three role variants with the linked source
selector held fixed. Reuse the old threshold, recalibrate ridge on
strict proof errors from the 30 development source pairs, or require
ridge and definition banks to choose the same declared role and
calibrate the weaker standardized margin. Role calibration assumes
the correct sources in development and retains at most one adverse
group empirically. The test scores source substitutions and role
errors separately.

Rerun one positive request per held group and clue view from raw
phrases through live encoding, catalog ranking, source and role
decisions, and serialization. The component timer uses resident
indices and excludes index build, data fetching, HTTP and queueing.

## Results

The merged catalog has 2,011 papers. The held data have 89
positive and 89 negative requests across 30 source groups. Each
clue view includes 356 cases with one designated source omitted.
The first count column measures distinct paper groups; all proof
counts measure requests.

| Clue | Source rule | Correct source pair / 30 | Correct author on positive / 89 | Strict proof / 89 | Wrong proofs when both sources are present | Wrong proofs with a source held out |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Excerpt | Linked | 30 | 78 | 57 | 40 | 0 |
| Excerpt | Anchored | 29 | 76 | 55 | 40 | 0 |
| Excerpt | Individual or paired | 30 | 78 | 57 | 40 | 0 |
| Keyword | Linked | 26 | 69 | 50 | 34 | 4 |
| Keyword | Anchored | 21 | 55 | 41 | 24 | 0 |
| Keyword | Individual or paired | 26 | 69 | 50 | 34 | 4 |

At keyword depth six, both correct papers are reachable for 26
groups. The linked baseline recovers all 26, so the learned pair
score has no headroom there for higher source recall. On held-out
source cases, the learned models retain the linked baseline's four
unsupported keyword results from a single substituted group.
Anchoring suppresses those outputs and gives up five correct source
groups among present keyword requests.

With linked source selection fixed, requiring stronger role proof
has the following consequences when both target articles are present.
Wrong counts cover source errors, wrong authors, or wrong quoted roles.

| Role rule | Excerpt strict / wrong | Keyword strict / wrong |
| --- | ---: | ---: |
| Original ridge cutoff | 57 / 40 | 50 / 34 |
| Recalibrate on strict proofs | 21 / 4 | 20 / 4 |
| Calibrated bank agreement | 15 / 1 | 13 / 1 |

Role screening moves the coverage–error trade-off substantially.
It leaves a large gap to the oracle role control, which returns
89 exact positive excerpt proofs and 78 keyword proofs given the
linked source proposals. An owner-only metric would miss many of
the citation failures. The 21 excerpt requests and 19 keyword
requests separating owner and strict counts are the next priority
for interpretation and evidence diagnostics.

The two live local timings ran on an RTX 4090 on ws1. Linked P95
is 39.3–39.8 ms for excerpts and 17.2–18.5 ms for keywords;
paired confirmation is 41.6–41.7 ms and 18.7–19.3 ms.
Both jobs in array 47082 agree on all quality and work counts.
Source counts reproduce the pilot array 47077. The evaluation uses
constructed clues and resolved publisher identities. It measures
a first-query component on a catalog, with no claim about
organizational experts or operational risk. [Aggregate results](results.json).

## Reproduce

Restore the two eLife projections and the frozen panels from the
[proof-group study](../engineering-proof-group-calibration-v1).
Run under Slurm with a GPU and a fresh output path:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.held_open_proof_search \
  CONFIRMATION_PROJECTION EXTENDED_PROJECTION \
  benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json \
  benchmarks/engineering-proof-group-calibration-v1/results.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  QWEN_EMBEDDING NEW_RESULTS
```

Repeat with `--reverse` and a distinct output path.
