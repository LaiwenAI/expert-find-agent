# Confirm source identity and author duties independently

Start from [the two-pair backfill experiment](../plos-joint-source-selection-v1/README.md). Reuse its 55-paper snapshot, 80 matched requests, two phrase inventories, two source clue types, source masks, and the original matching role gate. The first proposal is the same linked source pair and set of authors that the backfill path would return. The benchmark checks that all baseline support and error counts reproduce. All confirmation arms inspect that proposal only, and a rejection produces no recommendation from the rejected candidate set.

Cross four source checks with two role checks. Source checks use the weaker side of the proposed pair for acceptance. The fast checks use shared embedding similarity or unweighted lexical coverage. The extra model check uses the open Qwen3-Reranker-0.6B with the instruction to recover the specific paper described by the clue. It receives only the indexed title and first part of the abstract. The original last-two-sentence result clues are kept outside that input. Each source method rejects scores at or below the maximum incorrect-proposal score from the earlier, disjoint [source-confirmation panel](../plos-source-confirmation-v1/README.md). Calibration includes the first two linked pairs with and without their intended sources available and counts each intended-clue/proposed-pair tuple once per clue type.

Role agreement retains only proposed authors also approved by the [separate role-instruction ridge head](../plos-external-shared-role-gate-v1/README.md). It uses the same frozen embedding model, the original 84 training phrases under a role-specific retrieval instruction, and a positive declared-role margin. No calibration uses the 55 test papers. Report retained supported queries, wrong-source returns and unsupported returns on correctly identified sources. A supported return and an unsupported return can coexist when the accepted candidate set includes more than one author.

Timing reconstructs an executed cascade from measured stages. A query with a proposal is charged for its selected source confirmation, and for role agreement only if the source passes. It includes neither model loading nor index construction or service transport. Repeated variants are correlated. The panel's author relationships and duties are publisher-declared, and the evaluation contract requires identifying the intended papers.

## RTX 5090 results

Slurm job 42560 reproduces the two-pair baseline in every setting. Source calibration yields 41 exact source proposals and ten wrong proposals. At the strict cutoffs, dense, lexical and cross-encoder checks retain 21, one and 19 exact calibration proposals respectively.

| Confirmation | Supported, both present | Wrong sources, both present | Wrong authors, both present | Wrong sources, one missing | Result-clue P95 range |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 93 | 12 | 17 | 5 | 9.07–9.09 ms |
| Role agreement only | 81 | 10 | 14 | 5 | 17.59–17.64 ms |
| Dense source only | 40 | 1 | 7 | 4 | 9.07–9.09 ms |
| Lexical source only | 0 | 0 | 0 | 0 | 9.07–9.12 ms |
| Cross-encoder source only | 53 | 5 | 10 | 0 | 20.87–20.94 ms |
| Cross-encoder and role agreement | 47 | 3 | 8 | 0 | 29.28–29.40 ms |

The both-present counts pool 320 evaluations, including 160 with publisher-declared matches. The single-source masks pool 640 evaluations over the same four article pairs and wording conditions. Ranges use one per-wording P95 for the result clues with both sources present.

On result clues alone, cross-encoder confirmation preserves all 53 supported responses and reduces wrong-source responses from nine to five. On keywords it rejects all 40 supported responses alongside three wrong-source and seven wrong-author responses. Role agreement alone loses 12 supported responses while removing three wrong-author responses and two wrong-source responses overall. It leaves the five masked false proposals unchanged. These are conditional error reductions under an unchanged proposal generator. The zero masked returns for the cross-encoder accompany complete abstention on the keyword slice and do not establish a population risk guarantee.

Retain the cheaper proposal path as the baseline. A decision based on source evidence should report clue-specific coverage and wrong-source risk. The role-agreement result does not support paying for a second encoding pass by default. This pilot provides a fixed-threshold transfer check across publisher snapshots, with no expert-fit judgments beyond the structured role task. [Aggregate results](results.json)

## Reproduce

Use the construction and frozen inputs for [source-pair backfill](../plos-joint-source-selection-v1/README.md) and the original development corpus and panel from [source confirmation](../plos-source-confirmation-v1/README.md). Use the two pinned Qwen models and run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.confirm_source_and_role \
  DEVELOPMENT_PROJECTION CALIBRATION_PANEL HOLDOUT_PROJECTION HOLDOUT_PANEL \
  NISO_EXAMPLES BOUNDARY_PHRASES ROLE_TRAINING RIDGE_CALIBRATION_RESULTS BACKFILL_RESULTS \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
