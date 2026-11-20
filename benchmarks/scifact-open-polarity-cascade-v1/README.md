# Allocate 4B verification under a fixed batch budget

Use the same 263 calibration and 195 held-out human-labeled
claim–abstract pairs, 1,800-character evidence prefix, frozen readers
and calibrated margins as the
[polarity gate study](../scifact-open-polarity-gate-v1).
Check that both original calibrated readers reproduce on both panels.
Each evidence item is tested against opposing requested relations.
An opposite acceptance costs one unit of utility, a correct
acceptance earns one, and abstention contributes zero.

## Budget comparison

Start with the compact contrastive reader. In each consecutive batch
of up to 16 evidence items, send at most one quarter or one half to
the 4B reader. Floor the number of allowed upgrades within each batch.
For an upgraded item, use the 4B result. Both readers preserve
mutual exclusion. Compare these priorities:

- **Hash control:** frozen hash of claim and document IDs.
- **Uncertainty:** smallest absolute support–contradiction score gap.
- **Dual confidence:** largest of the lower scores for the two
  opposing requests, to detect broad acceptance by the binary model.
- **Learned value:** fit a ridge predictor of the 4B minus 0.6B
  utility gain on calibration. Its only inference features are
  the two compact scores, their minimum, and their absolute gap.

The equal-budget simulation uses the same precomputed margins.
Separately execute all-small, all-large and the uncertainty,
learned-value and dual-confidence cascades at the half budget.
Dual confidence was added to live timing after the initial
recorded-margin comparison. Compute the large
reader only on routed items. Time tokenization and inference
for both models, routing, and decision assembly from an already
available evidence batch. Model loading, retrieval, inter-request
queueing and networking are outside this timer. P95 is descriptive
across thirteen batches. Each execution begins after both
checkpoints have been used to reproduce reference decisions.
Compare live decisions with the recorded-margin replay to detect
batch-shape sensitivity.

The coverage and risk counts are conditional on a judged stance.
They do not quantify unknown evidence, general query traffic,
expert identity or final recommendation performance.

## Equal-budget replay

Counts on the same 195 held-out items. Each cell is correct-only /
opposite-only. No arm approves both directions.

| Upgrade priority | 48 upgrades | 97 upgrades |
| --- | ---: | ---: |
| Hash control | 52 / 25 | 66 / 27 |
| Uncertainty | 50 / 30 | 68 / 35 |
| Dual confidence | 50 / 22 | 67 / 15 |
| Learned value | 47 / 29 | 67 / 36 |

At half budget, dual confidence has roughly the same correct
recovery as the other priorities and fewer opposite acceptances.
The learned priority and smallest-gap uncertainty each accept
more opposite evidence than the hash control. A high minimum
score across opposing requests detects a behavior that a small
score gap can also exhibit when both requests are scored low.
This interpretation is specific to the tested readers and panel.

## Executed verification batches on RTX 5090

Both tasks of Slurm array `44739` on `ws5` reproduced the live
decision counts and both simulated budgets. P95 ranges span the two
runs with reversed arm order.

| Arm | Actual 4B calls (items) | Correct / opposite only | Results changed vs. score replay | Batch P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| All 0.6B | 0 | 32 / 24 | 0 | 196.0–196.4 |
| All 4B | 195 | 99 / 23 | 0 | 655.3–656.1 |
| Uncertainty | 97 | 68 / 35 | 0 | 502.4–502.5 |
| Learned value | 97 | 67 / 36 | 0 | 502.5–502.7 |
| Dual confidence | 97 | 65 / 15 | 2 | 473.1–474.1 |

Dual confidence used 97 of 195 possible 4B item evaluations.
Compared with all 4B it shortened measured batch latency and
reduced opposite approvals from 23 to 15, with 34 fewer correct
approvals. Two of its live decisions differed from the score
replay when the 4B reader handled only the routed subset.
Its observed recovery therefore does not meet a same-quality
replacement claim. The timing cannot establish service P95.

[Aggregate results and timings](results.json).

## Reproduce

Use the pinned projection, checkpoints and aggregate reference from
the polarity gate study. Repeat on a Slurm GPU with
`--order-offset 1` to reverse timed arm order.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.polarity_cascade \
  PROJECTION QWEN_RERANKER QWEN_4B_READER \
  benchmarks/scifact-open-polarity-gate-v1/results.json NEW_RESULTS
```
