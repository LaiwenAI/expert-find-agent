# Reuse and allocate whole-note author-verification calls

Freeze first-source selection, the existing character gate and Qwen3-4B. Use the stricter whole-note threshold from the [previous attribution intervention](../plos-owner-scope-intervention-v1/README.md). This replay keeps the 197-paper catalog and 72 previously used requests. It measures what can be recovered by reusing existing verdicts and changing which author pairs receive additional calls.

A call covers two authors. The original cascade sends a pair to the reader if at least one of its authors is undecided. The reuse route uses the resulting scores for both undecided and provisionally accepted authors and applies the stricter threshold. It keeps the character rejections and accepts unverified character positives as before. Cross score reuse with the original and stricter thresholds in a two-by-two design to separate the effect of reusing verdicts from stricter acceptance. The source and prompts stay fixed.

For each request, choose zero, one or two additional author pairs from those with at least one automatic acceptance and no existing call. Compare alphabetical pair order with priority given to the smallest character acceptance margin in the pair. At each cap, the compared routes receive the same number of additional calls on each request. Retain the original cascade and the full-note reader for every pair as controls. An oracle enumerates all values of the one-pair action and uses the evaluation labels solely to count how often an additional call could improve an outcome over reuse.

A clean success requires at least one recommended author and support for every recommended author. Report clean success, any supported author, and wrong-author and wrong-source recommendations. Report factored retrieval and verifier P95 and the exact call count. The full-note scores are collected once per pair in the replay. Each route is charged only for its actual calls. Timing excludes transport, provisioning and concurrent queues. Paired intervals resample author-name components.

## RTX 5090 results

Slurm jobs 42686 and 42689 ran on ws5. The second run adds the crossed controls and reproduces every earlier decision count. The stricter whole-note threshold was 17.75, frozen from the previous study, versus 14.5 in the original cascade.

| Clue | Author route | Any supported author | Clean success | Wrong author, correct source | Wrong source, withheld | 4B calls, present | Factored P95, present |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Excerpt | Original cascade | 57 | 37 | 22 | 0 | 103 | 100.30 ms |
| Excerpt | Stricter threshold | 56 | 46 | 13 | 0 | 103 | 100.30 ms |
| Excerpt | Reuse, original threshold | 56 | 36 | 22 | 0 | 103 | 100.30 ms |
| Excerpt | Reuse, stricter threshold | 53 | 44 | 13 | 0 | 103 | 100.30 ms |
| Excerpt | One extra pair | 49 | 43 | 9 | 0 | 136 | 105.30 ms |
| Excerpt | Two extra pairs | 46 | 40 | 7 | 0 | 146 | 109.98 ms |
| Excerpt | Full note for every pair | 48 | 41 | 9 | 0 | 188 | 142.40 ms |
| Keywords | Original cascade | 42 | 27 | 17 | 4 | 81 | 99.98 ms |
| Keywords | Stricter threshold | 41 | 33 | 11 | 4 | 81 | 99.98 ms |
| Keywords | Reuse, original threshold | 42 | 27 | 17 | 4 | 81 | 99.98 ms |
| Keywords | Reuse, stricter threshold | 39 | 32 | 11 | 4 | 81 | 99.98 ms |
| Keywords | One extra pair | 35 | 31 | 7 | 3 | 107 | 104.87 ms |
| Keywords | Two extra pairs | 32 | 28 | 5 | 3 | 116 | 109.50 ms |
| Keywords | Full note for every pair | 35 | 31 | 6 | 3 | 158 | 142.00 ms |

The crossed comparison attributes the improvement to the stricter threshold for already undecided authors. With exactly the original calls, clean successes rise from 37 to 46 for excerpts and from 27 to 33 for keywords. The descriptive paired 95% intervals are +5.56 to +20.55 and +2.78 to +15.07 percentage points. Any-supported counts each fall by one. Reusing the other score in an already paid pair does not improve clean success at either threshold. The extra-call routes start from the stricter reuse arm and trade further reductions in wrong-author queries for lower coverage.

The fixed-order and margin-priority routes have the same aggregate outcome counts and exact per-request call budgets. For the one-pair cap they choose different pairs on only two present queries per clue. Even an oracle choosing one extra pair improves at most two clean successes per clue over stricter reuse. This oracle considers only the stated candidate pairs and the same frozen reader. It does not upper-bound additional evidence sources or improved readers. Source-withheld results expose a different error class, with four keyword wrong-source recommendations under threshold-only and three under the additional-call routes. The observed benefit is an attribution calibration effect on a reused diagnostic set. [Aggregate results](results.json)

## Reproduce

Restore the publisher captures and checkpoints using the [body transfer benchmark](../plos-body-transfer-v1/README.md). Run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.legacy_selective_recheck \
    OLD_SNAPSHOT FIRST_TRANSFER FRESH_TRANSFER QWEN_EMBEDDING QWEN4B \
    OPEN_ATTRIBUTION_RESULTS BODY_CONFIRMATION_RESULTS CASCADE_RESULTS \
    SCOPE_INTERVENTION_RESULTS NEW_RESULTS
```
