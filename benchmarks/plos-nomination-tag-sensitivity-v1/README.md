# Stress-test early role nomination against incomplete and incorrect topic tags

This analysis uses the frozen [PLOS nomination review](../plos-nomination-review-v1)
and its 46 requests. It reuses the same role declarations, ranked recommendations
and topic–article judgment unit. Publisher-tagged articles may also be judged
irrelevant by a human. No review judgment is used here.

A topic–article pair contributes the difference in its citation count between two
methods. Sharing a pair in both arms cancels common slots, even across the two
requests for a topic. One relevance judgment applies to all citations of that
pair. Only 39 of the 58 pairs can change any of the six paired contrasts across
either declaration mask. Of those, 28 carry the publisher tag and 11 lack it.

With a fixed assignment to tagged pairs, the lowest possible difference includes
every negative-weight untagged pair and the highest includes every positive-weight
one. To make the lower bound cross below a margin, reverse enough tagged pairs with
positive weight, largest first. To make the upper bound reach a margin, reverse
tagged pairs with negative weight, largest magnitude first. These counts are exact
minimum numbers of topic–article judgments, with untagged judgments freely varying.

| Indexed declarations | Comparison | All tags correct | All pairs unjudged | Fewest tagged reversals to change the stated possibility |
| --- | --- | ---: | ---: | --- |
| 78 of 78 | Whole-ranked person four vs topic four | +15 to +27 | 0 to +27 | Six to make a gain below 2 pp possible |
| 78 of 78 | Bounded person four vs topic four | +7 to +13 | 0 to +13 | Two to make a gain below 2 pp possible |
| 78 of 78 | Bounded person four vs bounded union four | +3 to +5 | 0 to +5 | Two to make a tie possible |
| 78 of 78 | Bounded person four vs topic eight | −10 to −4 | −17 to +8 | One to make a loss of at most 1 pp possible |
| 78 of 78 | Whole-ranked person four vs topic eight | −2 to +10 | −17 to +22 | One to make a loss greater than 1 pp possible |
| 42 of 78 | Whole-ranked person four vs topic eight | −15 to −8 | −23 to +8 | Three to make a loss of at most 1 pp possible |

Bounds are net recommendation slots out of 230, not confidence intervals.
The +2 pp checkpoint needs at least five net slots. The −1 pp checkpoint allows
a loss of two slots. These are point-estimate sensitivity checks only. The risk,
latency and statistical uncertainty criteria still require separate evaluation.

At the matched four-source budget, whole-ranked person nomination can only gain
supported slots on this replay, under the shared topic judgment and fixed role
declarations. A strict gain still requires independent confirmation of the added
pairs. The bounded route's apparent shortfall against topic eight can change if
an influential tagged citation is rejected. Prioritize independent assessment of
the 39 contrast-bearing pairs for the paired claims. Review all 58 for absolute
quality or risk, using the existing independent reviewer forms. Neither result
generalizes to a new catalog or establishes enterprise fit.

[Aggregate bounds](results.json)

```sh
PYTHONPATH=src:. python -m experiments.credit.nomination_label_audit sensitivity \
  FROZEN_REVIEW_PACKET benchmarks/plos-nomination-review-v1/results.json NEW_RESULTS.json
```
