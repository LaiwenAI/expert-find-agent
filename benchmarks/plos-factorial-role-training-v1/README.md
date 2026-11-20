# Separate what is synthesized from how the role verifier is fitted

The [previous role-verification study](../plos-calibrated-role-contrasts-v1)
compared near-role cross entropy with single-clause ridge. That comparison
changed the training examples and objective together. Here each of the
three training sets is crossed with both fitting methods.

- All arms use a frozen Qwen3-Embedding-0.6B, the same 84 role
  descriptions, and 168 generated examples with identical per-role
  counts. The [generator](../../experiments/credit/role_swap_training.py)
  builds single-clause controls and two kinds of matched swaps. Within a
  swap pair, the two duties and their order stay fixed while assignment to
  the subject and peer is exchanged. Related and unrelated swaps use the
  same templates.
- Ridge uses the original regularization. Cross entropy starts from the
  same-data ridge solution multiplied by ten and runs 120 full-batch
  AdamW steps at learning rate 0.002. The original reader remains a
  reference. Publisher markup determines author scope for every arm.
- Thresholds are selected separately on 151 cases from 32 calibration
  papers with a 10% observed wrong-role cap. The two test views reuse
  the same 127 cases from 22 held-out papers with different query wording.
  The source and wording hashes, model revisions, and decision counts
  for all five repeated arms match the original study. Intervals below
  resample whole source clusters and are descriptive.

## Held-out results

Slurm job `43616` ran on one RTX 5090 on ws5. Each cell gives accepted positives / accepted same-author wrong roles /
clean cases. A clean case accepts the positive and rejects its rival.

| Training examples | Fit | Straightforward | Boundary |
| --- | --- | ---: | ---: |
| Single clause | Ridge | 127 / 0 / 127 | 92 / 5 / 87 |
| Single clause | Cross entropy | 127 / 0 / 127 | 93 / 5 / 88 |
| Unrelated swaps | Ridge | 127 / 0 / 127 | 95 / 15 / 82 |
| Unrelated swaps | Cross entropy | 127 / 0 / 127 | 96 / 12 / 87 |
| Related swaps | Ridge | 127 / 2 / 125 | 112 / 15 / 99 |
| Related swaps | Cross entropy | 127 / 0 / 127 | 100 / 7 / 96 |
| Recalibrated reranker | Frozen | 62 / 5 / 57 | 42 / 10 / 34 |

On boundary wording, related swaps improve clean decisions over the
single-clause control for both objectives. With ridge the increase is
12 cases (+9.45 points, 95% interval +5.15 to +14.18). With cross entropy
it is eight cases (+6.30 points, +1.82 to +10.56). Against unrelated
swaps, which control for the two-clause template, the gains are +13.39
points (+8.57 to +18.42) with ridge and +7.09 points (+2.44 to +11.90)
with cross entropy. Wrong-role acceptances are unchanged with ridge
and fall by five with cross entropy (−3.94 points, −7.04 to −1.25).

Within the related-swap dataset, switching from ridge to cross entropy
accepts 12 fewer positives and eight fewer wrong roles. The clean-case
interaction between related swaps and cross entropy relative to the
single-clause control is −3.15 points (−9.00 to +2.27). The two
related-swap heads also have 120 and 122 correctly ordered positive–rival
pairs before thresholding, compared with 122 for both single-clause heads.
The effects concern decisions at the independently calibrated operating
point. The observed boundary wrong-role rates are 15/127 (11.8%) with
ridge and 7/127 (5.5%) with cross entropy. A 10% calibration cap should
therefore not be treated as a test-risk guarantee. Boundary P95 is
6.92 ms for the shared query embedding and all six heads together and
8.47 ms for the reader. These component measurements exclude retrieval
and ownership parsing.

The comparison uses an already explored test panel and gives author scope
to both methods. It tests semantic role verification, not open-corpus
expert retrieval, enterprise transfer, or the synthesis of ownership
labels from unstructured text. A new collection should vary ownership
and competing duty independently, include independently worded cases,
and measure wrong-person and wrong-role rates under the same threshold
policy.

## Reproduce

Use the XML snapshots, pinned NISO examples, and open checkpoints from
the [calibrated role study](../plos-calibrated-role-contrasts-v1).
Run in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.factorial_role_training \
    OLD_PUBLISHER_SNAPSHOT LEGACY_TRANSFER_SNAPSHOT MODERN_DEV_XML MODERN_TEST_XML \
    benchmarks PINNED_NISO_EXAMPLES.json QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS.json
```

[Aggregate results](results.json)
