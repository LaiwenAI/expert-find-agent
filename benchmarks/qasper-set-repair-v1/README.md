# Learning small repairs to incomplete evidence sets

The [QASPER evidence-gap audit](../qasper-evidence-gap-audit-v1/README.md) and [frozen joint selector](../qasper-listwise-evidence-v1/README.md) show room for improvement once the evidence has entered the candidate pool. This experiment learns to replace one or two paragraphs from the pointwise output while holding the 16-paragraph shortlist, five output slots, and frozen open-model reranker fixed. It adds no model forwards to the retrieval path.

Start with the five highest fused scores. Enumerate all five-paragraph sets reachable with up to two swaps. A full 16-paragraph pool yields 56 sets within one swap and 606 sets within two, including the original output. Give a set a positive label if it contains any complete feasible annotator set. For each comparable training query, sample 64 positive–negative set pairs and learn L2-regularized preferences from feature differences. Earlier [marginal-gain](../qasper-final-evidence-sets-v1/README.md) and [paragraph-relation](../qasper-section-relation-graph-v1/README.md) experiments learned passage or edge scores. This experiment fits the complete-set endpoint directly.

| Route | Selection rule |
| --- | --- |
| Pointwise | Keep the five best fused scores |
| Score-only repair | Two-swap search with aggregate pointwise-score statistics |
| Joint one swap | One-swap search with score statistics, query coverage, semantic redundancy, sections and distances |
| Joint two swaps | Two-swap search with the same jointly trained score |
| Forced joint two | Accept any positive estimated improvement, without development calibration |

The official train split supplies fit labels. The disjoint development split calibrates the minimum predicted improvement, with fallback to pointwise among the options. Tie-break in favor of fewer edits. The previously frozen 137-question test panel is unchanged. All splits use one question per paper whose smallest available complete set contains two to five paragraphs. The label controls panel sampling, and the selection policy receives no gold labels. Compare to the diagnostic one- and two-swap ceilings to quantify which failures are repairable within the neighborhood.

The same live encoder and cross-encoder pass supplies all choices. Factored totals charge the measured shared prefix and selector operations, including feature construction. If calibration keeps the pointwise route, serving would skip the feature and selector work. The timings exclude training, offline source encoding, transport and service load. [SetR, ACL 2025](https://aclanthology.org/2025.acl-long.861/)

## RTX 5090 results

Job 41666 fits on 370 questions, calibrates on 112, and evaluates on the same 137 multi-evidence test questions as the previous studies. Of the training questions, 220 have both a complete and an incomplete set within the two-swap neighborhood. On test, 83 shortlists contain a complete set. One and two swaps make a complete set reachable for 66 and 78 questions, respectively, versus 43 complete pointwise outputs.

| Route | Complete / 137 | Gains / losses vs pointwise | Edited | Selector P95 | Factored total P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pointwise | 43 | 0 / 0 | 0 | <0.01 ms | 73.96 ms |
| Score-only repair | 43 | 0 / 0 | 0 | 0.58 ms | 74.42 ms |
| Joint one swap | 48 | 10 / 5 | 123 | 0.58 ms | 74.42 ms |
| Joint two swaps | 51 | 13 / 5 | 111 | 0.58 ms | 74.42 ms |
| Forced joint two | 50 | 13 / 6 | 126 | 0.58 ms | 74.42 ms |

The calibrated joint two-swap policy improves the complete count by eight. Its paired descriptive 95% interval is 0.00 to +11.68 percentage points, with a +5.84-point estimate. The policy uses threshold 0.10 and changes 111 of 137 outputs. The score-only control makes no changes. Joint selection remains below its hindsight ceiling and still loses five formerly complete results. The dataset provides the intended paper and paragraph evidence, without cross-paper expert relevance or individual attribution. [Aggregates](results.json)

## Reproduce

Use the official QASPER data and the same verified checkpoints as the [evidence-gap audit](../qasper-evidence-gap-audit-v1/README.md). The official train/dev archive supplies the pinned training JSON.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.qasper.set_repair \
  QASPER_DIRECTORY QASPER_TRAIN_DEV_ARCHIVE EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```
