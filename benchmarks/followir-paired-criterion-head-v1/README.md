# Learn criterion changes with a one-pass verification head

Reuse the exact paired judged subsets of the
[FollowIR threshold experiment](../followir-paired-criterion-gate-v1).
Freeze Qwen3-Reranker-0.6B, document windows, the standard prompt,
and original and revised conditions. Both baseline cutoff decisions
must reproduce on all three collections.

Replace the yes/no vocabulary projection with a linear head applied
to the normalized final-token hidden state. Each decision uses one
forward pass for the current query, conditions and document.

- **Verbalizer:** original model margin and transferred cutoff.
- **Pointwise head:** fit ridge regression to ±1 human relevance
  targets for the two views, using each view as a separate row.
- **Paired head:** use the same supervised rows and ridge penalty.
  Add a squared loss on the difference between the two scores for
  the same document. The target difference is −2 after invalidation
  and zero when the judgment stays the same.

Ridge penalty 0.001 and pair loss weight 1 are fixed for this
experiment. Fit the head on Core17 to score News21 for calibration
and on News21 to score Core17. For each route, take the maximum of the four per-class,
per-collection cutoffs computed to cap revised-criteria negative
acceptance at 10%. Then
fit each head jointly on Core17 and News21 and test with the frozen
cutoff on Robust04. Transfer of the cutoff between different fits
is empirical. Feature extraction runs on GPU, and the closed-form
ridge solve uses CPU. Qwen weights are frozen.

The pointwise head isolates the value of adaptation from the
value of the extra paired objective. Changes to training and
thresholding use only the Core17 and News21 annotations. The
Robust04 panel has already been used in prior experiments. It is
held out from these fits and cutoff calculations, and an independent
replication would still be needed for a confirmatory claim.

This benchmark measures per-document decisions. It does not time
the deployed head or test attribution to a named person. Both
views are evaluated for paired diagnostics. A deployed head would
only receive the current view.

## Results on RTX 5090

Both tasks of Slurm array `44714` on `ws5` reproduced all counts
and thresholds. The head uses 1,024 hidden features. The Robust04 comparison uses
the same 575 judged document–topic pairs as the threshold experiment.

| Route | Stable positives accepted after revision / 194 | Invalidated accepted after revision / 181 | Stable negatives accepted after revision / 200 | Invalidated accepted then rejected / 181 |
| --- | ---: | ---: | ---: | ---: |
| Original yes/no margin | 5 | 3 | 0 | 3 |
| Pointwise head | 5 | 1 | 0 | 4 |
| Paired head | 5 | 3 | 0 | 7 |

The criterion-pair loss changes which invalidated documents the model
accepts then rejects, but neither learned head improves stable
positive retention at its transferred cutoff. The pointwise head
accepts fewer known invalidated documents than the paired head,
with the same positive count. For both learned heads, the transferred
cutoff accepts zero stable positives in the out-of-domain Core17
calibration fold. This is a warning that sharing one cutoff across
collections and refitted heads can erase useful coverage.

These results do not support replacing the current verifier with
either learned head. The same-document intervention remains useful
for evaluation. A subsequent experiment should use unseen annotated
constraints to determine whether better input evidence or a model
that can adapt its internal representations improves the
coverage–risk trade-off.

[Aggregate results](results.json).

## Reproduce

Use the pinned snapshots and checkpoint from the threshold benchmark.
Run on a Slurm GPU with a fresh output path. Repeat with
`--order-offset 1` to reverse collection evaluation order.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.followir.paired_criterion_head \
  CORE17_PREPARED NEWS21_PREPARED ROBUST04_PREPARED QWEN_RERANKER \
  benchmarks/followir-paired-criterion-gate-v1/results.json NEW_RESULTS
```
