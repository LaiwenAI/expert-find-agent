# Sentence-level diagnosis of compact SciFact-Open verification

Use the [same 37 held-out claim–abstract pairs](../scifact-compact-open-v1/README.md), the frozen Qwen3-4B model, and the two existing thresholds. [SciFact-Open](https://aclanthology.org/2022.findings-emnlp.347/) provides sentence locations for these labels. Examine where the lexical selector leaves out an annotated sentence and whether restoring that sentence changes the reader's decision.

For the diagnostic intervention, replace the fewest compact-view sentences needed to include the full annotation. Drop the sentences farthest from the annotated locations, retain source order, and keep the sentence count fixed. This is possible for 35 of 37 pairs. Token counts may differ. A second diagnostic shows only the set of annotated sentences. Both interventions use the frozen compact cutoff. They use ground-truth locations to diagnose the reader.

| Gold label and annotation coverage | Pairs | Accepted as support, full | Accepted as support, compact |
| --- | ---: | ---: | ---: |
| Support, all annotated sentences present | 14 | 9 | 9 |
| Support, at least one annotated sentence missing | 12 | 3 | 1 |
| Contradiction, all annotated sentences present | 6 | 0 | 0 |
| Contradiction, at least one annotated sentence missing | 5 | 1 | 0 |

The full and compact runs reproduce the prior binary acceptance counts and pair-level transition counts. The compact view misses three true supports accepted by the full view. Two omit at least one annotated sentence and recover after the fixed-sentence-count replacement. The third already has all annotated sentences in view. The replacement also recovers one true support missed by both original views and removes no prior acceptance among eligible pairs. The annotated-only view accepts 9/26 supports and falsely accepts 1/11 contradictions.

These observations identify a selection failure in two cases and sensitivity to additional context or the reader's decision boundary in another. Replacing sentences can also remove distracting text. The annotation-guided intervention quantifies a diagnostic opportunity on these 37 pairs. Measuring attainable gains requires an annotation-free selector and a new evaluation set. [Aggregate results](results.json)

## Reproduce

Generate the pinned pair projection using the linked held-out study, then prepare the corresponding sentence annotations from the official SciFact-Open archive.

```sh
python -m experiments.scifact.compact_open_rationale prepare \
  OPEN_ARCHIVE FROZEN_PAIRS NEW_SENTENCE_LOCATIONS
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.compact_open_rationale run \
  NEW_SENTENCE_LOCATIONS FROZEN_PAIRS QWEN_4B_CHECKPOINT NEW_RESULTS
```

Job 41339 ran on an RTX 5090 on ws5.
