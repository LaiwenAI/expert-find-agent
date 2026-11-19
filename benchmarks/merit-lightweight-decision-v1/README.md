# Lightweight suitability scoring with disjoint profile groups

Re-evaluate MERIT with a split that joins pairs sharing either a normalized target title or the same multiset of normalized publication titles in the reviewer history. Split connected components as indivisible units. The prior title-only partition allowed 116 exact profile signatures to occur on multiple sides. The new split has 356 fit, 125 calibration, and 129 test pairs. The test has 45 positive pairs in 60 connected components. Profiles with different titles may still refer to the same person. The task is reviewer self-assessment on supplied paper-profile pairs; the full dataset has appeared in prior studies. [MERIT assessor task and earlier controls](../merit-evidence-dispatch-v1/README.md)

The lexical head uses Jaccard overlap across each profile and pooled target-term coverage. The dense maximum takes the best owned-publication cosine. The dense head combines lexical features with the maximum, second best, top-three mean, mean, spread and highest-score gap from frozen Qwen3-Embedding-0.6B. Its profile vectors are built offline. Only the target title and abstract are embedded at request time. Both heads use ridge-regularized logistic regression on the fit split.

The reader baselines use the same frozen Qwen3-4B-Instruct-2507 with the first two or all publications and the full target text. Select every decision threshold on calibration balanced accuracy, including reader thresholds. The dense gate routes small-margin decisions to the calibrated two-publication reader, with a 20% maximum escalation fraction in calibration. Requests execute each arm independently with rotating order and no saved query logits or embeddings. Test timing includes live feature computation, tokenization and any second pass. It excludes cold start and transport.

## RTX 5090 results on 129 test pairs

| Route | Accuracy | Balanced accuracy | Positive F1 | True positives / 45 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two-publication reader | 62.02% | 61.55% | 52.43% | 27 | 148.8 ms |
| Full reader | 69.77% | 69.56% | 61.39% | 31 | 218.3 ms |
| Lexical head | 65.12% | 58.25% | 41.56% | 16 | 0.15 ms |
| Best dense source | 71.32% | 60.95% | 39.34% | 12 | 7.56 ms |
| Dense and lexical head | 73.64% | 65.83% | 51.43% | 18 | 7.63 ms |
| Dense head with optional fallback | 73.64% | 65.83% | 51.43% | 18 | 7.58 ms |

The joint head reduces P95 approximately 19.5x relative to the two-publication reader. Its paired accuracy difference is +11.63 pp (95% connected-component bootstrap interval +2.00 to +21.62). Its positive recall is 40.00%, versus 60.00% for the two-publication reader and 68.89% for the full reader. The calibrated gate selects zero fallbacks, so the last arm duplicates the head's decisions. The full reader has the best positive F1 and balanced accuracy.

The timings assume resident profiles. Offline building of 1,424 source embeddings took 7.66 seconds. These results identify a fast first-stage candidate; they do not validate its use as a hard filter when relevant experts must be retained. The bootstrap operates on the connected components. The reused helper's generic sampling-unit label in the JSON refers to source papers. [Aggregates](results.json)

## Reproduce

Use a Slurm GPU allocation and the pinned MERIT release plus both verified open Qwen checkpoints:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.decision.merit_lightweight_decision \
  MERIT_ASSESSOR_TEST_JSON QWEN_EMBEDDING_CHECKPOINT QWEN_READER_CHECKPOINT NEW_RESULTS
```
