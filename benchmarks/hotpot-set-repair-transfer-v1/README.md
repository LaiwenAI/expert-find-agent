# Transferring evidence-set repair to cross-document sentence retrieval

The [QASPER repair experiment](../qasper-set-repair-v1/README.md) learned to replace up to two paragraphs in a frozen five-item ranking. This study tests its fit on an independent task whose annotated evidence spans at least two documents. Use the official [HotpotQA distractor development release](https://hotpotqa.github.io/) with its supplied documents and annotated supporting sentence locations. Each evidence key is the original title and sentence position, even when the same sentence text appears in multiple documents.

Build three HotpotQA panels by stable hashing of question IDs, selecting the test panel first. Reject any row sharing a normalized supplied article title, question or ID with a previously selected row. Require between two and five available annotated facts from at least two documents. Reserve 256 questions for internal test, 100 for calibration and 220 for target fitting. The title exclusions apply even to distractor documents, yielding a topic-mixed diagnostic subset of the public development set. The official hidden test is outside this evaluation.

Each route independently of the labels uses the same frozen query and source embeddings, dense Top-16 and cross-encoder scores. Source text includes the title and search units are sentences. Pick five positions, or all available positions if fewer. Complete evidence requires every labeled supporting sentence. The ranker's score statistics and evidence features use the same definitions as the QASPER experiment, treating each document title as a section. Context documents come from the benchmark rather than whole-corpus candidate retrieval.

| Route | Decision |
| --- | --- |
| Pointwise | Return the five highest fused scores |
| Two-document rule | If all five belong to one document, replace the last with the highest-scoring available sentence from a different title |
| QASPER transfer | Refit the frozen QASPER recipe on its official training panel and carry over its published threshold of 0.10; no HotpotQA training or calibration |
| Target score only | Fit on HotpotQA training cases and calibrate on its disjoint calibration panel, with score features only |
| Target joint | Use the same target fitting and calibration with set-level coverage and source relationship features |

The reranker pass is timed once per question for every route. Report factored P95 from the measured common prefix plus local feature and selector operations. This excludes offline sentence embeddings, service transport and concurrent load. No additional model forwards are used to select the final set.

## RTX 5090 results

Job 41675 uses 220 comparable preference-training questions in QASPER and 162 in HotpotQA. The 256-question HotpotQA test panel contains 209 bridge and 47 comparison questions. The fixed shortlist contains all supporting sentences on 193 questions. A complete set is within two swaps of pointwise on 192.

| Route | Complete / 256 | Gains / losses | Changed | Selector P95 | Factored total P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pointwise | 122 | 0 / 0 | 0 | 0.000 ms | 41.12 ms |
| Two-document rule | 122 | 0 / 0 | 5 | 0.002 ms | 41.12 ms |
| QASPER transfer | 102 | 6 / 26 | 169 | 0.505 ms | 41.56 ms |
| Target score only | 120 | 6 / 8 | 194 | 0.505 ms | 41.56 ms |
| Target joint | 132 | 19 / 9 | 229 | 0.504 ms | 41.56 ms |

Direct transfer loses 20 complete outputs (paired descriptive 95% interval −12.11 to −3.52 percentage points). Fitting and calibrating on separate HotpotQA panels recovers a net 10 (interval 0.00 to +8.20 points), while the target score-only control loses two. The target joint policy uses the same zero threshold selected on calibration and changes 229 of 256 outputs. It adds roughly 0.50 ms at the selection stage. This supports testing domain-specific training and monitoring loss of previously correct outputs. It leaves whole-corpus retrieval, claim interpretation and personal attribution untested. [Aggregate results](results.json)

## Reproduce

Use the official distractor JSON, the pinned QASPER train/dev archive and the same verified open checkpoints as the [QASPER experiment](../qasper-set-repair-v1/README.md). Run on a Slurm CUDA node.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.decision.hotpot_set_repair_transfer \
    QASPER_TRAIN_DEV_ARCHIVE HOTPOT_DEV_DISTRACTOR_JSON EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```
