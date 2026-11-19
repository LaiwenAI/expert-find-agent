# Certified reranking against fixed depth on QASPER full text

Take one question from each of 96 official test papers using fixed hash ordering. All arms receive the same known paper and its deduplicated paragraphs. Freeze Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B. Score a reranked paragraph as `0.75 * cosine + 0.25 * sigmoid(reranker margin)`. Exhaustive and fixed-depth routes evaluate the same contiguous blocks of eight in dense order. The certified route stops when the next paragraph's `0.75 * cosine + 0.25` upper bound is strictly below the current fifth fused score. Strict comparison preserves source-position tie breaking. Each evaluated block has the same scorer inputs regardless of route.

The comparison includes fixed depths of 8, 16 and 32, and the dense Top-5. The weights and depths were fixed for this diagnostic. Test papers have been reused for previous experiments. Evaluate evidence against human paragraph annotations on the 83 questions with at least one complete alignable answerable evidence set. Complete evidence means the returned five contain an entire reference set.

| Route | Exact exhaustive order / 96 | Mean reranker checks | Complete evidence | Evidence F1 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 10 | 0.0 | 55.42% | 26.49% | 6.98 ms |
| Fixed 8 | 59 | 8.0 | 61.45% | 29.14% | 42.97 ms |
| Fixed 16 | 87 | 15.9 | 68.67% | 31.72% | 74.71 ms |
| Fixed 32 | 94 | 30.0 | 69.88% | 32.07% | 132.09 ms |
| Exhaustive | 96 | 52.6 | 69.88% | 32.07% | 292.39 ms |
| Certified | 96 | 38.5 | 69.88% | 32.07% | 234.05 ms |

The certified route skips a block or more on 41 queries, saves 26.9% of reranker evaluations overall, and improves complete evidence over dense search by 14.46 percentage points (descriptive paper bootstrap interval +6.02 to +22.89). Against fixed 32, it preserves the exhaustive list on two additional queries but gains no measured paragraph quality and has 77% higher P95. Fixed 32 is the stronger efficiency control under the measured evidence metrics. The cost of guaranteed exhaustive output is a separate result. This is a single warm RTX 5090 run. Times include query embedding, resident paragraph scoring, reranker input construction and forwards. Cross-paper discovery, networking, evidence attribution and answer generation are outside the measured path.

Ranked list truncation and cross-encoder early exit have established SIGIR precedents. This test applies a score bound to preserve the output of one fusion rule. To support a contribution beyond this diagnostic, a joint expert and evidence stopping policy still needs to improve the measured frontier against strong truncation controls. [Ranked List Truncation, SIGIR 2024](https://doi.org/10.1145/3626772.3657864) · [Similarity-based Early Exit, SIGIR 2025](https://doi.org/10.1145/3726302.3729962)

## Reproduce

On a Slurm GPU compute node with the [official QASPER assets](https://aclanthology.org/2021.naacl-main.365/) and the pinned checkpoints, run:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.qasper.certified_reranking \
  DATA_DIRECTORY EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON
```

[Aggregate results](results.json) record model, corpus, script and raw output hashes. The release excludes paper text and per-question rows.
