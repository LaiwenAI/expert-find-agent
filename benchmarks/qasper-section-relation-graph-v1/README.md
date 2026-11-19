# Section-guided shortlists and paragraph relation scoring

Use the official QASPER development and test splits, with separate fit and calibration paper subsets established by the [previous depth protocol](../qasper-adaptive-depth-v1/README.md). The new comparison has two panels. The first reuses the 120 additional papers from the earlier evidence-set experiment. The second uses the other 56 official test papers outside that experiment's two panels and the original 96-paper shortlist pilot. Other studies have used the broader public QASPER corpus. [QASPER human evidence](https://aclanthology.org/2021.naacl-main.365/)

| Route | Paragraphs sent to the same frozen cross-encoder |
| --- | --- |
| Flat 16 | Best 16 dense passages |
| Section-guided 16 | Keep the first eight, then combine paragraph similarity with the section's best match and heading overlap, discounting already represented sections |
| Flat 32 | Best 32 dense passages |

Within each route, the pointwise control uses dense similarity and cross-encoder relevance. The learned node controller adds a fit-only estimate of whether an individual paragraph belongs to an annotated evidence set. The graph controller adds a second estimate for whether two paragraphs occur in the same annotated set. Edge features combine both paragraphs' pointwise signals, embedding similarity, lexical similarity, distance, common section, and the extra question words covered together. Beam search over sets of five compares the two terms in a common objective.

Choose node and edge weights on the calibration papers for complete evidence. Zero recovers the corresponding control. Also report the best strictly positive calibration edge weight as a diagnostic. Every route performs real query encoding and reranking. Each selector uses that route's identical measured prefix, adding its own feature and selection time. No graph work is charged for the pointwise path. The task supplies the paper and judges paragraph evidence. It does not test cross-paper expert discovery or personal attribution.

Hierarchical allocation and set-wise evidence retrieval are established research directions. This experiment isolates a small learned relation score with matched query, source, and reranker conditions. [SetR, ACL 2025](https://aclanthology.org/2025.acl-long.861/) | [BEAR, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1348/)

## Results on RTX 5090

| Panel | Route | Complete shortlist | Pointwise complete | Forced-graph complete | Pointwise P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Additional, 105 evaluable | Flat 16 | 95 | 79 | 78 | 83.2 ms |
| Additional, 105 evaluable | Section-guided 16 | 95 | 79 | 78 | 89.0 ms |
| Additional, 105 evaluable | Flat 32 | 101 | 80 | 79 | 142.8 ms |
| Remaining, 48 evaluable | Flat 16 | 43 | 34 | 33 | 76.3 ms |
| Remaining, 48 evaluable | Section-guided 16 | 43 | 35 | 33 | 76.3 ms |
| Remaining, 48 evaluable | Flat 32 | 47 | 34 | 33 | 122.0 ms |

The fit data provided 5,072 nodes and 74,279 edges, only 290 of which occurred together in an annotated evidence set. Calibration set both learned weights to zero at every depth. The strictly positive diagnostic lost complete evidence under all tested settings. The flat 16 and flat 32 routes return the same number of complete sets on the remaining papers, with a 37.5% P95 reduction for 16. Their output sets need not match, and their candidate coverage differs. Timing and quality are for the known-paper retrieval task; matched depth controls paragraph count rather than the number of tokens. [Full aggregates](results.json)

## Reproduce

On a Slurm CUDA node with the frozen QASPER and open-weight checkpoints:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.qasper.section_relation_graph \
  QASPER_DIRECTORY EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```
