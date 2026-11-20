# Decide when to read reviewer histories after compact ranking

This replay tests whether a low-cost policy can use source evidence
to correct a strong first-pass ranker. Each [LR-Bench
comparison](https://aclanthology.org/2026.acl-long.2035/) asks for
one winner from two supplied choices. The paper-centric direction
is primary. A paper is the query and the options are two reviewers
with different human self-reported familiarity grades. The reverse
direction ranks papers for a fixed reviewer. These grades address
reviewer familiarity at the profile level. They do not adjudicate
which individual source supports a claim.

Reuse the two `RATE-0.6B` scores from the
[matched-profile comparison](../../docs/merit-suitability-card.md#matched-size-profile-comparison).
The compact route compares two released keyword profiles. Reading
then scores the owned titles and abstracts and takes the top-three
mean for each option, using the same checkpoint and pooling as the
compact route. No new embedding or reranker inference is performed
in this replay. The official released RATE predictions serve as a
separate published baseline.

## Train and evaluate the routing policy

Hash the anchor ID into five folds. For each held-out fold, also
purge any training comparison that shares either option ID with
that fold. Thus, in the primary direction, neither a test paper ID
nor a test reviewer ID appears in its training comparisons. This
does not establish absence of shared prior publications or
near-duplicate papers across folds. Each direction learns its own
policy. Score scales, thresholds and mixture weights use only that
fold's retained training comparisons. Option order cannot change
the routing features.

- **Margin router.** Retrieve history scores when the absolute
  compact margin is below a training quantile. Either replace the
  compact score or combine it with the source score. Fit mixture
  weights over a fixed grid using all training rows or just the
  training rows the margin policy would route.
- **Utility router.** Fit a ridge model on the training label
  “source ranking correct minus compact ranking correct.” Its
  features are the absolute margin, mean compact score, smaller
  log history size, absolute log size difference and history
  overlap. Rank by predicted repair advantage and use the frozen
  training quantile to decide when to use the source score.
- **Full-history control and hindsight ceiling.** Compare reading
  every history with a label-informed oracle that repairs a
  mistake only when the source route can fix it, spending the same
  aggregate source scoring budget. The oracle takes the cheapest
  eligible repairs. It bounds routing of this evidence channel
  under the reported dot-product costs.

The nominal 10%, 25% and 50% thresholds use training data only.
Actual test acquisition counts can differ. A zero weight skips
the evidence computation. Report total source dot products as a
work proxy above the shared compact comparison. All candidate
histories are pre-encoded and exact top-three selection scores
every available document. The work count does not include
retrieval latency, service loads or monetary costs. Use the
official strict accuracy, rating-gap weighted loss and paired
anchor intervals.

## Results

Two tasks in Slurm array `44249` on ws5 reproduced the aggregate
byte for byte. The replay is CPU-only on top of the preceding
GPU-encoded scores. The final comparator set adds direct raw-score
switching and weights fitted on the routed training subset after an
initial exploratory replay. The folds and thresholds continue to be
fit independently of each test fold.

| Route | Paper selects reviewer / 212 | Source dots | Reviewer selects paper / 1,184 | Source dots |
| --- | ---: | ---: | ---: | ---: |
| Published RATE predictions | 163 | — | 891 | — |
| Local compact RATE-0.6B | 160 | 0 | 880 | 0 |
| Read every history | 150 | 2,424 | 895 | 12,290 |
| Margin router, raw score, nominal 25% | 159 | 598 | 894 | 3,102 |
| Margin router, conditional mix, nominal 25% | 157 | 244 | 893 | 3,102 |
| Margin router, raw score, nominal 50% | 155 | 1,156 | 895 | 5,928 |
| Utility router, raw score, nominal 25% | 160 | 541 | 882 | 3,602 |
| Utility router, raw score, nominal 50% | 161 | 951 | 883 | 6,494 |

Reading all histories on the primary panel repairs 15 mistakes
and creates 25. The 25% margin route captures nine repairs but
creates ten losses. Across the tested policies, the largest
observed primary improvement is one net correct comparison.
Its paired 95% anchor interval is −3.15 to +4.04 percentage
points. Given the cost of that policy, the hindsight oracle can
capture all 15 source-channel repairs. This illustrates an
opportunity to recognize **which** evidence can change a choice,
with little demonstrated discrimination from scores and history
metadata alone.

On the reverse panel, the 50% margin rule reaches the same strict
accuracy as reading every history with about 48% of the source
dot products. This is an observed efficiency point on a different
task direction. The paired difference between that route and
compact alone still includes zero. The experiment does not show
an improvement over the compact first pass in the primary task.
An independent test collection or matched endpoint with
adjudicated evidence is needed before using these policies as
quality controls in an expert search Agent.

[Full aggregate](results.json)

## Reproduce

Use the pinned release and first reproduce the RATE-0.6B files
from the linked matched-profile benchmark. Run the replay with
the release directory and the generated prediction directory.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m experiments.retrieval.lrbench_budgeted_evidence \
  LRBENCH_RELEASE MATCHED_PROFILE_PREDICTIONS NEW_RESULTS.json
```
