# Test nomination conclusions under missing publisher topic labels

Replay the [bounded nomination experiment](../plos-bounded-topic-v1) on
Slurm job `43536` using the same projections, checkpoints, calibration
grid, candidate methods and 46 queries. Record emitted source DOIs in
output order, retaining the same source more than once when it supports
multiple authors. Check that every route and coverage mask reproduces
the original totals for grounded and off-label citations.

Deduplicate every emitted citation into a topic–article judgment.
Two identical forms contain titles, abstracts and publisher URLs for
58 distinct pairs. Thirteen pairs lack the exact publisher topic tag.
The forms omit route, rank, tag status and score. Reviewers judge
substantive relevance to the topic. Contributor roles are held fixed
from the published declarations. The blank-form reconciliation
correctly returns 58 unresolved cases and no human judgments.

## Label sensitivity

Hold publisher-tagged positives fixed and let every untagged
topic–article pair be either relevant or irrelevant. For each two-route
contrast, count the net difference in emitted slots per pair. Common
slots cancel, and duplicate slots for a pair share one label. The lower
bound adds the negative unresolved weights to the tagged difference.
The upper bound adds the positive weights. These are exact extrema
conditional on the tagged positives and the observed outputs. They
contain no sampling uncertainty.

| Contrast, full declaration coverage | Tagged difference | Range over untagged labels |
| --- | ---: | ---: |
| Bounded person four vs topic four | +7 | +7 to +13 |
| Bounded person four vs bounded union four | +3 | +3 to +5 |
| Bounded person four vs topic eight | −5 | −10 to −4 |
| Whole-ranked person four vs topic eight | +3 | −2 to +10 |
| Bounded person four with pilot-gated cutoff vs topic four | −1 | −1 to +3 |

All values are recommendation slots out of 230. The bounded route's
best-case deficit versus topic eight is 4 slots (1.74 percentage
points), beyond the existing 1-point noninferiority margin even at the
point-estimate level. Under roughly half declaration coverage, this
deficit is at least 9 slots. Resolving untagged cases alone cannot
rescue that comparison. Whole-ranked nomination is more sensitive to
the pending labels. It still requires risk and latency comparisons.
Avoid using these labels to retune on this replay catalog.

A later human analysis should allow reviewers to challenge tagged
positives as well. Reconciliation counts agreed decisive labels
and retains unresolved slots, including disagreements, for
adjudication. It also reports paired bounds that cancel shared
unresolved cases. A new catalog is needed to validate a revised policy
and to estimate generalization.

[Aggregate replay and bounds](results.json)

## Reproduce

Run the bounded nomination replay with `--witness-output NEW_WITNESSES`.
Then generate and score the blinded packet:

```sh
PYTHONPATH=src:. python -m experiments.credit.nomination_label_audit prepare \
  HOLDOUT_PROJECTION REPLAY_RESULTS NEW_WITNESSES benchmarks/plos-bounded-topic-v1/results.json PACKET_DIR
PYTHONPATH=src:. python -m experiments.credit.nomination_label_audit reconcile \
  PACKET_DIR/packet.json REVIEWER_A REVIEWER_B NEW_REPORT
```
