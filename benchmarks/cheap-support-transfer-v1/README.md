# Transfer a lightweight evidence score to two eLife catalogs

Earlier work found that enumerating all compatible source pairs can
expose all label-supported plans in the candidate pool. The resulting
allocation still depends on how the agent ranks those plans.
Compare a cheap learned score against the existing joint plan and
against the all-edge rank-weighted planner. All policies retain depth
128, the same verified contributor-role index, a budget of three
distinct declarations, and the same final verifier and proof order.

Fit one weighted ridge regression on the original 130-article PLOS
fixture. A training row is a unique topic–document pair from a
compatible two-source proof candidate. The target is presence of a
publisher topic tag. Use equal total weights for observed-tag and
unjudged rows, an unpenalized intercept, and L2 penalty 1. Unjudged
rows act as zeros only in this proxy training objective. The six
features are saturating transforms of the hybrid, dense and lexical
ranks, topic-token coverage in the title and abstract, and lexical
match presence. Test uses the two original eLife projections and
derives the same queries as the frozen exact-index benchmark. No
eLife topic judgments are consumed in scoring or tuning.

| Route | Read allocation |
| --- | --- |
| Role joint | Keep the established same-contributor plan. |
| Rank set | Score the existing all-edge candidate plans with lexical/dense rank fusion weights and the existing alternative-pair bonus. |
| Learned set | Reuse the full candidate-plan frontier. Select the plan with the largest weakest-clause linear score over any compatible proof in that plan. Break ties in favor of the established plan, then by source IDs. |

Publisher tags score the test after verification. Report whether
a supported proof exists in the retrieved pool, survives the three
reads, and is returned. The matched quota and common-issuance views
measure different subsets. The former uses citation ranks to equalize
the number of returns on each panel. The latter uses exactly the
queries where both policies return. Missing-tag bounds and paired
topic-cluster intervals are separate. Timing measures each route's
local work with the embedding already resident. It excludes network,
serving queues, and training. This is a proxy-label transfer test
for evidence selection on previously benchmarked eLife catalogs.

## Results

Slurm job 47579 completed on ws3 with an RTX 5090. Both traversal
orders reproduce the fit, the original joint results, and all new
decisions. The fit uses 675 topic–document rows with 128 observed
positive labels from 160 PLOS requests.

| Held eLife panel | Label-supported in the pool | Role joint | Rank set | Learned set |
| --- | ---: | ---: | ---: | ---: |
| Extended (205 requests) | 82 | 59 / 119 | 60 / 119 | 56 / 119 |
| Confirmation (193 requests) | 84 | 80 / 141 | 83 / 138 | 80 / 138 |

Cells give supported proofs over natural issuances. Each arm returns a supported proof whenever its read set contains one. The
observed shortfall on the extended panel occurs before final proof
ranking. At the shared quotas of 119 and 138, the rank-set route
changes the observed support count by +1 and +3 relative to joint.
Its difference under missing tags can be −3 to +4 and −4 to +7,
respectively. Keep the simpler joint planner. The learned route
changes three experts in the extended panel, with three tagged
proofs lost and no gains. On the confirmation panel it matches the
observed tagged count. Across both panels the learned route's warm
P95 is 14.71–14.96 ms, compared with 14.48–14.65 ms for joint.
The test leaves independent expert suitability and enterprise
qualification judgments open.

[Both replay receipts](results.json)

## Reproduce

Restore the PLOS XML from the original
[130-source fixture](../plos-credit-cross-source-v1), then use the
two frozen eLife projections and encoder specified in the
[exact-index benchmark](../linked-role-lane-index-v1#reproduce).
Run in a Slurm CUDA allocation and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.cheap_support_transfer \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  EXTENDED_PROJECTION benchmarks/elife-extended-role-holdout-v1/manifest.json \
  CONFIRMATION_PROJECTION benchmarks/elife-source-guard-confirmation-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/linked-role-lane-index-v1/results.json NEW_RESULTS
```
