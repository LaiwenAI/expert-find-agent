# Fine-tune a one-pass stance verifier with low-rank attention updates

Use the frozen 263 training/calibration and 195 held-out judged
pairs from the [scientific polarity gate](../scifact-open-polarity-gate-v1).
The query and 1,800-character abstract prefix share the neutral
stance prompt used by the [previous ridge readout](../scifact-open-polarity-readout-v1).
Compare two single-forward Qwen3-Reranker-0.6B routes with the same
classification objective and training schedule:

- **Head only:** freeze the entire backbone and train a two-class
  linear head on the normalized terminal hidden state.
- **LoRA:** initialize the same head and attach rank-8 updates to the
  query and value projections of the final four attention layers.
  Train the new parameters and keep the pretrained weights frozen.

Train with AdamW, cross entropy, a learning rate of 0.0005, gradient
norm clipping at 1, eight items per training batch, and four fixed
epochs. Use the same seeds, per-epoch orders, and grouped folds as
controls between methods. LoRA and head parameters train on the
Slurm GPU. Calibrate two nonnegative cutoffs from out-of-fold
predictions, targeting at most 10% opposite-stance approval per
direction. Grouped folds share neither claim nor abstract IDs.
Retrain on all 263 items with the frozen schedule and evaluate the
transferred cutoffs on the 195 items. Do not fit or tune a threshold
on the test row labels. Abstention is a thresholded decision; this
study does not use independently labeled unknown examples.

Latency covers tokenization, one GPU pass with unmerged adapters
where applicable, and output assembly for batches of up to 16.
It excludes retrieval, queues, training and startup, and is
descriptive over 13 batches. This public cohort has been used in
earlier exploration and remains held out from all fits here.

## Results on RTX 5090

Both tasks of Slurm array `44767` on `ws5` reproduced the final
training loss summaries, cutoffs and outcome counts. One LoRA fold's
final-epoch training loss varied by 0.00027 across runs. Both traces
are retained.

| Route | Trainable parameters | Correct supports | Correct contradictions | Opposite approvals | Batch P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear head only | 2,050 | 3 / 102 | 2 / 93 | 13 | 117.9–118.5 |
| Head + LoRA | 165,890 | 0 / 102 | 12 / 93 | 16 | 119.4–119.7 |

[Aggregate results](results.json).

## Threshold diagnostics

With zero thresholds, the LoRA classifier correctly assigns 103
of 195 stance labels versus 96 for the head-only control. It
recovers 29 contradictions instead of 3, while correct supports
fall from 93 to 74. This operating point forces nearly every
item into a direction and is diagnostic only.

The transferred risk cutoffs change that trade-off substantially.
LoRA accepts 12 annotated contradictions and zero annotated
supports, and approves a wrong direction on 16 held-out items.
The head-only control accepts 2 annotated contradictions and 3
supports, with 13 opposite approvals. In particular, LoRA
misclassifies 16 of 102 support cases as contradictions even though
its out-of-fold false approvals were within calibration limits.
These results show a change in polarity behavior, without enough
correct coverage or stable risk to replace the 4B reader. The
training set has only 45 distinct claims and the evaluation
covers this single evidence relation.

## Reproduce

Use the pinned projection, compact checkpoint and original
aggregate reference. Run on a Slurm GPU and repeat with
`--order-offset 1` to reverse training and evaluation arm order.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.polarity_lora \
  PROJECTION QWEN_RERANKER \
  benchmarks/scifact-open-polarity-gate-v1/results.json NEW_RESULTS
```
