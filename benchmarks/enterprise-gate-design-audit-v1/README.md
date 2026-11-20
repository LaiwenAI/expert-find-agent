# Audit the finite-sample design of the sealed enterprise gates

Use only the [public frozen protocol](../enterprise-sealed-first-query-v1/protocol.json).
No requests, arm outcomes, or reviewer labels enter this audit. It
separates what the current study can establish from design targets
for an independently sampled replication. The original gates and
sealed cohort remain in force.

## Absolute coverage and adverse-risk gate

The frozen scorer uses two exact one-sided binomial bounds with
2.5% allocated to each endpoint. An unresolved issued group counts
as adverse. With 123 groups:

| Adverse or unresolved issued groups | Minimum groups issuing a recommendation |
| ---: | ---: |
| 0 | 85 |
| 1 | 110 |
| 2 or more | No attainable passing count |

The coverage condition alone would require at least 85 issued groups.
Even with every group issuing, the 5% upper risk cap permits only one
adverse or unresolved group. The audit verifies the boundary counts
against the existing scorer.

For a planning scenario, let the issued count follow a binomial
distribution. Conditional on issuance, let each group's adverse or
unresolved event have the specified probability. Sum each admissible
issued count's probability multiplied by the conditional probability
that its adverse count meets the original bound.

| Independent groups | 75% true coverage, 1% conditional risk | 75% true coverage, 2% conditional risk | 75% true coverage, 3% conditional risk |
| ---: | ---: | ---: | ---: |
| 123 | 37.2% | 14.5% | 5.6% |
| 250 | 87.8% | 48.2% | 18.5% |
| 500 | 99.8% | 85.9% | 42.2% |
| 1,000 | >99.9% | 99.5% | 76.2% |
| 2,000 | >99.9% | >99.9% | 97.5% |

These are single-arm pass probabilities. Passing both arms requires
their joint distribution. One arm's probability can only put an
upper bound on the two-arm event.

## Paired supported P@5

Use a separate scenario model where each independent group contributes
a loss of one supported slot, no difference, or a gain of one.
Five slots per group give a 20-point P@5 change for the two nonzero
outcomes. The discordance probability is the chance that the methods'
counts differ. Conditional on the number of discordant bootstrap
draws, the number of wins is binomial. This gives the exact
empirical resampling CDF for this scenario. It approximates the
production gate, whose percentile comes from 4,000 seeded draws.

| Independent groups | True +3 pp, 20% discordance | True +3 pp, 40% discordance | Equal quality, 20% discordance |
| ---: | ---: | ---: | ---: |
| 123 | 25.6% | 14.5% | 27.1% |
| 250 | 44.0% | 23.7% | 43.2% |
| 500 | 75.4% | 43.1% | 72.4% |
| 1,000 | 96.6% | 72.1% | 94.6% |
| 2,000 | >99.9% | 95.1% | 99.9% |

The first two columns estimate the chance that the paired lower
bound exceeds +2 points. The third estimates the chance that its
lower bound reaches the −1 point noninferiority limit. The paired
and absolute scenarios do not imply a joint pass probability.
Latency, clustering above the frozen group, and multiple-slot
changes are outside these calculations.

## Interpretation

Complete the already frozen blind review and report its exact paired
intervals and counts. A failed gate can reflect an incompatible result
or low power near the target effect. Report which bounds fail and by
how much. The scenarios motivate a distinct follow-up design sized
from plausible coverage, unresolved risk, and paired disagreement.
They do not estimate current effectiveness or justify selecting a
larger cohort after opening the sealed results.

[Full scenario grid](results.json)

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_gate_design_audit \
  benchmarks/enterprise-sealed-first-query-v1/protocol.json NEW_RESULTS.json
```
