# Learning a lightweight complementary-evidence selector on QASPER

The [dense shortlist experiment](../qasper-completion-retrieval-v1) recovered 68 reachable test cases with four candidates, of which the frozen reranker completed 42. This experiment holds that candidate pool fixed and tests whether a small selector trained on completing paragraphs can improve accuracy, reduce latency, or both.

One question per paper is selected by the same rule as the earlier experiments. Reveal all but one paragraph of a known human-annotated set. A hit requires adding a paragraph that completes any extractable reference. The question-only Qwen3-Embedding-0.6B retrieves four unseen paragraphs from the known paper. Compare its first candidate, a frozen anchor-conditioned Qwen3-Reranker-0.6B, and two fitted selectors.

The four-feature model uses dense cosine, reciprocal rank, question-term overlap, and paragraph vocabulary size. The eleven-feature model adds semantic similarity to already verified paragraphs, section and position relationships, and word coverage relative to the verified evidence. Both fit ridge regression on feature differences between every completing and non-completing candidate in each training shortlist. Every question that contains both types receives equal weight. The ridge penalty is fixed at 0.25 on scaled features. Both models fit on official training papers. Development and test papers are disjoint from training and from each other. Development and test questions were used in previous retrieval experiments.

The fitted selectors use the same embeddings as retrieval and require no cross-encoder call at inference. The frozen reranker receives the question plus up to 128 tokens from each verified paragraph. Component P95 sums per-query dense retrieval time and the observed selection time. Both fitted selectors are charged for extracting all eleven features. The frozen arm includes assembling the anchor prompt and cross-encoding four pairs. These are staged measurements on a GPU compute node and exclude index build, cross-paper search, queueing and final evidence verification.

## Results

Successful Slurm job 42740 used one RTX 5090 on ws5 for both Qwen models. The small selector fits a ridge system on CPU in that job. The two models fit 528 positive–negative comparisons from 176 of 370 eligible training questions. The development and test panels reproduce the previous panel hashes, reachable counts, and frozen-reranker completion counts.

| Selector | Dev completions / 112 | Dev component P95 (ms) | Test completions / 137 | Test component P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| First dense candidate | 22 | 7.31 | 33 | 7.35 |
| Frozen anchored cross-encoder | 29 | 34.90 | 42 | 42.56 |
| Learned question features | 22 | 8.22 | 34 | 8.25 |
| Learned complement features | 34 | 8.22 | 50 | 8.25 |
| Best possible in fixed shortlist | 53 | — | 68 | — |

The complement features add 12 dev completions and 16 test completions compared with fitting question features alone. The paired 95% intervals for the difference are +3.57 to +18.75 percentage points on dev and +4.38 to +18.98 on test. Against the frozen cross-encoder, the complement selector adds five and eight completions. Those intervals are −2.68 to +11.61 points and 0.00 to +11.68 points, respectively. The observed accuracy difference favors the fitted selector, with weaker evidence than the feature ablation. Test component P95 is 80.6% lower.

The training labels describe one deliberate evidence omission with known human anchors. The result motivates a fixed-budget test with predicted anchors and distractor sources, including cases with zero and multiple omissions. This study measures neither matching experts nor the decision to recommend or abstain. The fixed pool excludes all valid completions in 69 test cases. Selection improvements can address only the remaining 68.

[Aggregate results](results.json)

## Reproduce

Use the verified public QASPER v0.3 files and frozen checkpoint manifests from the [retrieval benchmark](../qasper-completion-retrieval-v1). The script requires the official training archive and Slurm CUDA.

```sh
PYTHONPATH=src:. python -m experiments.qasper.completion_selector \
  QASPER_DIRECTORY QASPER_TRAIN_DEV_ARCHIVE EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS.json
```
