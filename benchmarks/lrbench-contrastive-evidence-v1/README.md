# Compare two reviewers using the same publication evidence

Test source-level comparison on the 212 paper-centric pairs from
[LR-Bench](https://aclanthology.org/2026.acl-long.2035/).
Each pair has two supplied reviewers with different self-reported
familiarity grades. All ranking methods must choose between those
same reviewers. The labels test whole-profile familiarity. They
supply neither publication-level relevance judgments nor evidence
of personal responsibility for the work.

Use the
[pinned RATE-0.6B checkpoint and keyword profiles](../../docs/merit-suitability-card.md#matched-size-profile-comparison).
The compact and source routes share a query embedding. Nominate
at most two publications per reviewer using cosine against their
owned title-and-abstract vectors. Exclude normalized target-title
matches from that nomination. The expensive methods get exactly
the same source excerpts; reviewer identifiers and scores are
omitted from the comparison prompt. The available publication text
itself is unchanged.

| Arm | Scoring |
| --- | --- |
| Compact RATE-0.6B | One score per compact keyword profile. |
| Raw top-three | Mean of the three highest source cosines. |
| Pointwise reader | Apply the frozen Qwen3-Reranker-0.6B to each nominated abstract and use the maximum per candidate. |
| One-order comparison | Frozen Qwen3-4B-Instruct-2507 compares both evidence bundles in one prompt, with initial position assigned by a hash of the paper and unordered reviewer IDs. |
| Two-order agreement | Pass both display orders in one forward. Return the same winner only when both orders choose that reviewer. |
| Order-symmetric margin | Average the two signed reviewer logit differences for a forced-choice diagnostic. |
| Selective replacement | Invoke the two-order method below a compact-margin threshold, replacing the compact choice only on agreement. |

The selective 25% and 50% thresholds are computed on the training
part of each [identity-separated fold](../lrbench-budgeted-evidence-v1).
Both the paper ID and reviewer IDs of a held-out pair are excluded
from the training comparisons for its fold. The all-request variant
uses agreement on every comparison. The score fusion rules,
display order, and nomination depth are fixed; no score weights
or prompts are tuned against test ratings.

Report strict accuracy, rating-gap weighted loss, repairs and
losses against compact, and paired paper-anchor bootstrap
intervals. Repeated comparisons do not become independent labels
by varying the display order. Tie and abstention count as misses
for standalone choices. Selective replacement reuses the compact
choice on abstention.

Measure compact query embedding and scoring on every request.
Charge measured source nomination plus the appropriate timed
reader call to any route that reads evidence. The shared
preparation is timed once per request. Selective-route P95 is
assembled from the measured compact and two-order timings,
with no extra service or queueing effects. The timed model
forwards use three separate frozen open checkpoints.

## Results

Both RTX 5090 tasks in Slurm array `44254` on ws5 reproduced all
non-timing outputs. The panel has 150 paper anchors. The two-order
model agrees on a winner in 87 of 212 comparisons, choosing the
higher-rated reviewer in 81 of those. In 98 comparisons it chooses
different winners as the display order changes. Five comparisons
abstain in both orders. The remaining 22 abstain in one order.
One order presents 288,021 total input token slots and two orders
present 576,042. Each reader sees the same 839 source abstracts
when run on all pairs.

| Route | Correct / 212 | Repairs / losses versus compact | 4B comparisons | P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Published RATE | 163 | — | — | — |
| Compact RATE-0.6B | 160 | 0 / 0 | 0 | 7.91–8.08 |
| Raw top-three | 149 | 14 / 25 | 0 | 7.98–8.16 |
| Pointwise reader | 137 | 23 / 46 | 0 | 44.28–44.34 |
| One-order comparison | 138 | 26 / 48 | 212 | 91.49–92.88 |
| Order-symmetric margin | 156 | 27 / 31 | 212 | 176.93–178.67 |
| Agreement with compact fallback, nominal 25% | 162 | 2 / 0 | 49 | 168.56–171.61 |
| Agreement with compact fallback, nominal 50% | 168 | 9 / 1 | 102 | 174.44–174.47 |
| Agreement with compact fallback, all pairs | 168 | 9 / 1 | 212 | 176.93–178.67 |

The nominal 50% route gains 3.77 percentage points over the local
compact baseline, with a descriptive paired 95% anchor interval
of +0.98 to +7.07 points. Its comparison against the published
RATE predictions is +2.36 points with an interval of −2.02 to
+6.90 points. The top-line gain depends on retaining the compact
choice when comparison is inconclusive. Agreement catches some
true corrections even though the ungated judge is weaker than
the compact ranker. Invoking it for half the pairs halves most
of the additional model work versus the all-pair route. Its
request-level P95 stays on the expensive path. This pilot
motivates a stronger paired confirmation experiment on an
independent cohort, using a shorter evidence window or a small
open decision model to target latency. No source-level claim
or hard qualification is evaluated by LR-Bench.

[Both GPU aggregates](results.json)

## Reproduce

Use the pinned LR-Bench and RATE release assets from the linked
matched-profile benchmark. Keep the same local checkpoint layout
and start the replay in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.lrbench_contrastive_evidence \
  LRBENCH_RELEASE NEW_RESULTS --order-offset 0
```
