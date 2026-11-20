# Learn a compact, one-pass polarity decision

Reuse the [scientific polarity benchmark](../scifact-open-polarity-gate-v1)
and its pinned evidence, reader checkpoints and held-out split. The
compact route uses one neutral prompt for the entire claim–abstract
pair. A linear ridge readout scores support versus contradiction from
the normalized final-token representation of frozen
Qwen3-Reranker-0.6B. There is no language-model output projection
on this route. Two nonnegative cutoffs turn the signed score into
support, contradiction or abstention. No explicit unknown examples
are used to train the head.

Fit the readout with ridge penalty 0.001 on the 263 calibration
pairs. First obtain out-of-fold margins from four grouped folds:
claims that share an abstract remain together. Use those margins to
choose a per-direction cutoff allowing at most 10% observed
opposite-stance acceptance. Refit the head on all calibration items
and apply these fixed cutoffs to the 195 held-out pairs. Claim IDs and
abstract IDs do not cross the outer split. This is a reused test panel
held out from this fit, not a new independent evaluation cohort.

Compare the one-pass route with the established two-pass compact
verifier and one-pass 4B ternary reader. Both reuse their previously
calibrated cutoffs and must reproduce their published decisions in
aggregate. Time complete warm verification of batches of up to 16
items after a warm-up batch for each arm. Timed work includes
tokenization, GPU processing and the compact readout. It excludes
training, checkpoint loading, retrieval, arrival queue, and transport.
The experiments test evidence stance on known labels. A low margin
leads to abstention, which does not establish detection of genuinely
insufficient evidence.

## Results on RTX 5090

Both tasks of Slurm array `44759` on `ws5` reproduced the
decisions, cutoffs and grouped fold sizes. Batch P95 ranges span
two runs in reversed method order.

| Route | Correct only | Opposite only | Abstained | Batch P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| One-pass compact readout | 7 | 8 | 180 | 117.1–117.5 |
| Two-pass compact contrast | 32 | 24 | 139 | 193.9–194.9 |
| One-pass 4B ternary | 99 | 23 | 73 | 647.4–649.5 |

The one-pass compact route lowers the warm component time, but
returns no correct contradiction on 93 human-annotated
contradictions. The supervised readout therefore fails the
quality requirement for replacing either baseline. The grouped
out-of-fold calibration likewise yields only 23 correct versus
25 opposite acceptances. On this panel the failure is visible
even before evaluating the outer test set.

[Aggregate results](results.json).

## Reproduce

Use the pinned projection, checkpoints and aggregate reference from
the polarity gate study. Run on a Slurm GPU and repeat with
`--order-offset 1` to reverse timed arm order.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.polarity_readout \
  PROJECTION QWEN_RERANKER QWEN_4B_READER \
  benchmarks/scifact-open-polarity-gate-v1/results.json NEW_RESULTS
```
