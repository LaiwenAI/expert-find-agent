# Calibrate and train near-role verification on disjoint publisher sources

The [frozen cross-format transfer](../plos-frozen-scope-role-transfer-v1)
found both a calibration shift and poorer score separation under
phrases that mention a competing responsibility. Test the two
factors independently. Use the same target author's positive and
near-role evidence fields as in the transfer study. Contributor
ownership is given by the publisher tree, and the wrong-owner foil
is removed for all methods before semantic verification.

- **Reader** uses the pinned Qwen3-Reranker-0.6B and the action-head
  prompt from the frozen transfer experiment.
- **Single clause**, **other-role swaps**, and **near-role swaps**
  train linear ridge heads over the same frozen Qwen3-Embedding-0.6B.
  Each uses the same 84 seed descriptions and 168 generated examples.
  The [existing swap builder](../../experiments/credit/role_swap_training.py)
  holds two duties fixed and changes the author to whom each duty is
  assigned. The single-clause and unrelated-role controls separate
  adding compositional examples from choosing related foils.
- **Near cross entropy** and **near rank loss** start with the same
  near-swap ridge weights and run the same number of optimizer steps
  on the same data. The rank-loss arm adds a logistic loss that scores
  the assigned role above the role attributed to the other person.

Calibration uses role contrasts from the earlier modern PLOS capture,
with any source or normalized byline overlap with the held-out cases
removed. The calibration phrases come from the pinned NISO glossary
examples. The two test banks use the same benchmark queries as the
frozen transfer experiment. Exact normalized wording does not
overlap among the training descriptions, calibration examples, and
the two test banks. Paraphrases can share meanings, and each bank
reuses its small set of phrases across papers.

Select each threshold on calibration by maximizing correctly accepted
positives with no accepted same-author foil in that case, subject to a
10% calibration cap on wrong-role acceptances. Freeze all thresholds
before evaluating the held-out articles. Report both ordered pairs
before thresholding and clean decisions after it. Match the acceptance
unit and evidence windows across methods. The reader scores two role
fields per query. The cheaper heads each consume one query embedding;
reported head P95 measures a shared forward with all experimental
heads computed together, as a conservative online measurement under
this benchmark's execution conditions. Timing excludes index access
and ownership parsing.

## Results

Slurm job 43115 ran on one RTX 5090 on ws5. The calibration split
has 151 cases from 32 papers. The 127 held-out cases span 22 papers.
Each cell lists accepted positives / accepted foils / clean cases.

| Model | Calibration | Straightforward test | Boundary test | Boundary P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Reader, recalibrated | 101 / 12 / 89 | 62 / 5 / 57 | 42 / 10 / 34 | 8.86 |
| Single clause | 120 / 4 / 119 | 127 / 0 / 127 | 92 / 5 / 87 | 7.19 |
| Other-role swaps | 124 / 6 / 121 | 127 / 0 / 127 | 95 / 15 / 82 | 7.19 |
| Near-role swaps | 132 / 11 / 121 | 127 / 2 / 125 | 112 / 15 / 99 | 7.19 |
| Near cross entropy | 121 / 4 / 120 | 127 / 0 / 127 | 100 / 7 / 96 | 7.19 |
| Near rank loss | 131 / 9 / 122 | 127 / 6 / 121 | 112 / 13 / 101 | 7.19 |

The independently recalibrated reader accepts 62 and 42 correct
roles compared with 33 and 29 at the legacy thresholds, and
accepts five and ten foils. The risk-constrained decision policy
should account for both outcomes.

For the ridge heads, matching the synthetic data count, near-role
swaps recover more difficult positive roles than the single-clause
control. They also accept ten additional foils, bringing held-out
foil acceptance above the 10% calibration cap. The near
cross-entropy arm makes nine more clean boundary decisions than
the single-clause ridge head (author-cluster interval +2.29 to
+12.61 percentage points), accepting two more foils. This comparison
changes both the training data and optimization. The matched
near-rank-loss ablation gains five clean boundary cases over near
cross entropy (−1.05 to +8.82 points) and accepts six additional
foils in each test view. Threshold-independent correct ordering
on boundary wording is 122/127 for single clause, near cross
entropy, and near rank loss.

The fixed-window comparison supports independent calibration and
shows a trade-off between coverage and same-author semantic errors.
The experiment reuses the previously reported external query panel.
These component times exclude search. A new panel with independently
authored questions is needed before selecting a production policy.

[Aggregate results](results.json)

## Reproduce

Use the pinned legacy and modern XML snapshots, NISO example file and
open checkpoints. The dev XML comes from
[the previous transfer capture](../plos-credit-transfer-v1), and the
test XML comes from [the external capture](../plos-credit-external-v1).
Run in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.calibrated_role_contrasts \
    OLD_PUBLISHER_SNAPSHOT LEGACY_TRANSFER_SNAPSHOT MODERN_DEV_XML MODERN_TEST_XML \
    benchmarks PINNED_NISO_EXAMPLES.json QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS.json
```
