# Semantic shortlists for completing a QASPER evidence set

This experiment tests the candidate stage after the agent has verified partial evidence. The paper is given. Show all but one paragraph of a complete human-annotated evidence set, and consider any unseen paragraph that completes a known set correct. Use the official QASPER v0.3 development and test splits, with disjoint papers and the previous one-question-per-paper sampling rule. The test panel repeats the controlled [remaining-evidence benchmark](../qasper-remaining-evidence-v1). Both splits have been used for previous experiments, and no parameters are fit here.

## Comparison

Each arm offers the same frozen Qwen3-Reranker-0.6B exactly the same anchored prompt and up to four or eight candidate paragraphs. Compare six ways of picking the pool:

| Candidate order | Signals |
| --- | --- |
| Lexical | Binary overlap between question and candidate, weighted by paper-local inverse document frequency |
| Gap | Lexical overlap, doubling query terms absent from every revealed paragraph |
| Dense | Frozen Qwen3-Embedding-0.6B retrieval with the question alone |
| Anchored | Same embedding checkpoint, with a missing-evidence instruction, question and excerpts of verified evidence |
| Residual | Question embedding with its positive projection onto the normalized centroid of verified paragraphs removed |
| Hybrid | Reciprocal-rank fusion of anchored and gap rankings with a fixed constant of ten |

All dense routes reuse the same section-plus-paragraph vectors. Both anchored prompts use up to 128 tokens per verified paragraph. Anchored retrieval changes the query instruction as well as its input, so its contrast against question-only dense retrieval measures the combined design. No labels enter the rankers after choosing the controlled query and anchor view.

Reachable means at least one completing paragraph survives the shortlist. Completed means the reranker selects one. Paired bootstrap intervals resample papers. Hypotheses across methods and budgets are exploratory. The timing sums observed retrieval and reranking times per question to estimate a serial component P95. It includes lexical feature extraction or query encoding as applicable. It excludes construction of the anchor prompt, offline indexing and request handling. Offline encoding time and vector storage are reported separately.

## Results

Slurm job 42723 ran on one RTX 5090 on ws5. The development and test panels contain 112 and 137 papers. Each cell reports reachable / completed questions, with separate test timing.

| Candidate order | Dev top 4 | Test top 4 | Test top 4 P95 (ms) | Dev top 8 | Test top 8 | Test top 8 P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexical | 28 / 13 | 36 / 25 | 40.2 | 51 / 23 | 57 / 26 | 79.4 |
| Gap | 24 / 11 | 34 / 23 | 40.4 | 51 / 22 | 57 / 26 | 79.9 |
| Dense | 53 / 29 | 68 / 42 | 42.5 | 73 / 30 | 88 / 37 | 81.7 |
| Anchored | 61 / 26 | 73 / 36 | 47.0 | 82 / 27 | 90 / 39 | 84.8 |
| Residual | 22 / 14 | 27 / 20 | 42.8 | 34 / 17 | 44 / 28 | 82.5 |
| Hybrid | 51 / 20 | 65 / 35 | 47.9 | 80 / 28 | 95 / 37 | 85.8 |

- **Question-only dense dominates lexical gap filtering at four reads in this setup.** On test, it doubles reachable cases, from 34 to 68, and completes 42 instead of 23. The completion gain is 13.9 percentage points (paired 95% interval +5.1 to +21.9). The corresponding dev gain is 16.1 points (+8.0 to +24.1). On test, serial component P95 rises from 40.4 to 42.5 ms. Reuse of an existing dense index makes this an inference-time comparison. The test index holds 6,765 vectors in 27.7 MB and took 19.2 seconds to encode.
- **A broader shortlist alone does not ensure better selection.** At eight test reads the hybrid exposes 95 completing passages by question, compared with 88 for dense alone, and both finish 37 cases. Question-only dense even completes fewer test cases at eight than at four. At four, the anchored query reaches five more test questions but finishes six fewer, with both differences uncertain. Removing the visible-anchor projection reduces reachability sharply in both splits. Keep question-only dense as the four-read reference and freeze its pool to test training the selector to reward complementary support.
- **Scope.** This experiment starts with a known paper and gold anchors. A subsequent evaluation needs predicted anchors and competing sources. [QASPER-MP](https://github.com/debobanerjee/LC-EBG/tree/main/datasets/QASPER-MP) offers a published paper-identity stress test with ten distractors per gold paper, but it derives from the same QASPER development split. [EfficientRAG](https://aclanthology.org/2024.emnlp-main.199/) provides a trained iterative retrieval comparison for independent multi-hop experiments.

[Aggregate results](results.json)

## Reproduce

Use the same verified QASPER v0.3 files and frozen checkpoints described in the [original shortlisting benchmark](../qasper-remaining-evidence-v1). Prepare a local file digest manifest for both Qwen checkpoints and run on Slurm CUDA.

```sh
PYTHONPATH=src:. python -m experiments.qasper.completion_retrieval \
  QASPER_DIRECTORY EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS.json
```
