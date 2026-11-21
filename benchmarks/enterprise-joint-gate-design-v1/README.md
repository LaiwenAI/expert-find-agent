# Plan a paired enterprise trial that must pass all quality and risk gates

The [original design audit](../enterprise-gate-design-audit-v1)
calculates a single arm's coverage and risk pass probability and the
paired quality pass probability separately. This extension estimates
how often the candidate and comparator pass both absolute gates and
the paired quality gate in the **same** trial. Its only study input is
the public [frozen protocol](../enterprise-sealed-first-query-v1/protocol.json).
No enterprise requests, search outputs, or review labels are used.

Each independently sampled organization or near-duplicate group has
one of three states per arm: abstained, issued with adverse or
unresolved evidence, or issued with all requirements supported.
Both arms have 75% issuance. Conditional adverse risk varies from
1% to 3%. The paired arms draw a common state for 90% or 100% of
groups and independently otherwise. An issued group with unresolved
evidence contributes to the adverse-risk gate. In the planning
model, only clean groups contribute supported slots.

A clean arm versus a non-clean arm gains one supported slot. Among
two clean arms, spread the remaining wins and losses to attain the
assumed +3-point mean gain for superiority, or zero for
noninferiority, at 20% or 40% paired discordance. The resulting model
has at most a one-slot difference per group. Neither the dependence
of paired decisions nor the magnitude of their differences is
estimated from the candidate frame.

The simulation uses 100,000 trials per scenario and the same exact
binomial cutoffs and exact paired percentile-bootstrap analogue as
the original design audit. The last approximates the frozen
4,000-draw scorer. Program checks compare all three marginal pass
rates with their analytical values. The worst-case Monte Carlo 95%
half-width is 0.31 percentage points. The joint probability bounds
in the JSON use only the three analytical marginal rates and make
no assumption about dependence *between pass events*.

## Results

Slurm job 47532 completed on ws6. A workstation replay with the same
seed reproduced the rates within Monte Carlo error. The table shows
the probability of passing both arms' absolute gates and the indicated
paired quality gate, assuming 2% risk and a 90% shared state fraction.

| Independent groups | Superiority, 20% discordance | Superiority, 40% | Noninferiority, 20% | Noninferiority, 40% |
| ---: | ---: | ---: | ---: | ---: |
| 355 | 41.0% | 22.0% | 34.7% | 19.8% |
| 500 | 62.4% | 35.8% | 59.7% | 36.1% |
| 750 | 87.0% | 57.1% | 82.9% | 55.9% |
| 1,000 | 95.9% | 71.6% | 93.9% | 71.1% |
| 1,500 | 99.7% | 87.9% | 99.1% | 86.8% |
| 2,000 | >99.9% | 95.1% | 99.9% | 94.5% |

Even if all [355 provisional candidates](../enterprise-regional-frame-v3)
were eligible and independent, this scenario would give a low chance
of passing the gates. A planning target of 80% for each gate combination
first appears in the reported grid at 750 groups with 20% discordance
or 1,500 with 40%. At 3% risk and 40% discordance, the 1,500-group
probabilities decline to 78.6% and 77.6%. These are scenario guideposts
for a new cohort. A separate pilot must estimate issuance, adherence,
paired discordance, and effective group counts before locking that
cohort. The noninferiority calculation covers only the quality and
risk prerequisites for a cost claim. It does not model the additional
latency and cost tests.

[Full results and analytic dependence-free bounds](results.json)

## Reproduce

Use NumPy and SciPy in the research environment.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_joint_gate_design \
  benchmarks/enterprise-sealed-first-query-v1/protocol.json NEW_RESULTS.json
```
