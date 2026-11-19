# Locate retrieval and verifier failures with annotated evidence interventions

Use the 500 official NLI4CT test statements with supplied report sections and line-level annotations. Reuse the [frozen two-expansion selector and frozen Qwen3-4B reader](../nli4ct-adaptive-reader-v1/README.md). The live baseline sees the same excerpts as before. A second view shows only assessor-marked lines, in their original trial and order.

An equal-line-count intervention is feasible for 43 of the 236 queries missing an annotated line. For both intervention arms, preserve all selected annotated lines, replace the same non-annotated lines within each trial, and leave the query unchanged. One arm adds the missing annotated lines. The control adds the same number of previously unselected, unannotated lines. Greedily match line length and original position to the missing line. The annotations determine interventions only, never the serving baseline. Unannotated lines can still contain valid evidence. Equal line counts do not guarantee equal token lengths.

Every arm uses the same reader and prompt. Starts rotate across statements. Report forced binary correctness and the outcome allowing an insufficient-evidence response. Uncertainty intervals resample connected components of shared trials. Timings cover prompt construction, tokenization and inference on the selected view. They exclude selection and report discovery. This is a diagnostic for an annotated evidence task, not an evaluation of a medical decision system or expert ranking.

## RTX 5090 results

| Panel | Reader input | Correct | Correct with abstention | Abstentions |
| --- | --- | ---: | ---: | ---: |
| All | Fixed two | 322 / 500 | 256 | 98 |
| All | Annotated lines only | 330 / 500 | 279 | 78 |
| Fixed two contains all annotated lines | Fixed two | 175 / 264 | 155 | 32 |
| Fixed two contains all annotated lines | Annotated lines only | 177 / 264 | 155 | 34 |
| Fixed two misses annotated lines | Fixed two | 147 / 236 | 101 | 66 |
| Fixed two misses annotated lines | Annotated lines only | 153 / 236 | 124 | 44 |
| Equal-size intervention | Fixed two | 23 / 43 | 21 | 6 |
| Equal-size intervention | Swap in annotated lines | 23 / 43 | 22 | 2 |
| Equal-size intervention | Swap in unannotated lines | 24 / 43 | 22 | 7 |

Job 41996 reproduced the original fixed-two coverage and correctness counts. On the full panel, annotated-only inputs fixed 30 errors and introduced 22, for +1.60 percentage points in forced binary accuracy (paired component-bootstrap 95% interval −0.92 to +4.00 points). The reader still missed 170 statements with assessor-marked lines supplied. On the 43-case matched subset, annotated swaps use an average of 477.95 tokens, versus 478.53 for the control. The forced binary difference versus the control is −2.33 percentage points (95% interval −11.90 to +5.88). Both return the same number of correct judgments with abstention allowed, but precision and coverage can trade off. In this subset annotated swaps reduce abstentions from seven to two compared with control.

The equal-size subset is selective. Most missing-evidence cases could not accommodate a matched replacement within their per-report budgets. Annotated evidence can be incomplete or require context. Neither intervention isolates a unique explanation for every error. The full test and the matched comparison show that optimizing line coverage alone is an inadequate proxy for verifier accuracy in this configuration. [Aggregate results](results.json)

Run on a Slurm CUDA node with the pinned public archive, frozen controller, and local open-weight checkpoint from the linked reader study.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.nli4ct_evidence_intervention \
    NLI4CT_ARCHIVE FROZEN_CONTROLLER READER_CHECKPOINT NEW_RESULTS
```
