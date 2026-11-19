# Section pruning on human-annotated full-text evidence

The [official QASPER test split](https://aclanthology.org/2021.naacl-main.365/) contains 1,451 questions over 416 research papers. Each method receives the intended paper. The replay deduplicates exact paragraph texts while preserving their first source position and section. It encodes the question and section-prefixed paragraphs with frozen Qwen3-Embedding-0.6B, capped at 512 tokens on input. Every method uses the same embeddings. Human labels are used only for evaluation, with no fitting or routing threshold selection.

- **Flat.** Score every paragraph and return the first five by score and source position.
- **Two-section beam.** Open the two sections with highest centroid scores.
- **Section beam filled to five.** Open those sections and continue down the centroid order if needed to reach five candidates.
- **Section upper bounds.** For each section with centroid (c) and maximum member distance (r), the best paragraph score is at most `dot(q, c) + norm(q) * r`. Evaluate sections in descending bound order. Stop when every unopened bound is strictly below the fifth precise score. Compute in float64 with outward slack. A check against the flat result verifies the entire ordered output.

These are shallow vector controls. [PageIndex](https://github.com/VectifyAI/PageIndex) uses language-model navigation over a document tree, a distinct method.

The complete-evidence metric counts a question when the retrieved paragraphs include every text paragraph in at least one nonempty, alignable human annotation. It covers 1,256 questions in 402 papers. Annotations involving unaligned passages or no evidence are excluded from this metric. Pixel and table retrieval are outside the experiment. CPU timing includes resident paragraph scoring, routing and sorting. It excludes document discovery, query encoding, content I/O and answer generation. It reports the P95 of five per-question timing observations summarized by their median.

## Results

| Route | Same order as dense / 1,451 | Mean paragraphs scored | Complete evidence | Evidence F1 | CPU P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat | 1,451 | 46.24 | 65.21% | 29.13% | 8.71 us |
| Two-section beam | 133 | 6.15 | 52.39% | 29.31% | 8.36 us |
| Section beam filled to five | 241 | 8.23 | 59.24% | 27.02% | 10.73 us |
| Section upper bounds | 1,451 | 42.07 | 65.21% | 29.13% | 43.17 us |

Fixed two sections return fewer than five paragraphs on 702 questions. With both routes required to return five, the centroid beam loses 5.97 percentage points of complete evidence versus flat search (paper-cluster interval -8.48 to -3.46). The upper bounds skip at least one paragraph on 1,358 questions but reduce the mean count by only 9%. Traversal costs more than the dense matrix operation at this paper size.

Slurm job `40637` uses RTX 5090 for embeddings and executes the timed score/routing stage on the allocated CPU. Offline encoding covers 19,710 paragraphs in 57.59 seconds, followed by 0.098 seconds to compile section metadata from the vectors. The findings apply to exact paragraph search inside a known paper. Author attribution, cross-source navigation, and end-to-end service timing require separate evaluation. [Aggregate results](results.json)

## Reproduce

Extract the [official QASPER test assets](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz) into a local data directory. On a Slurm GPU compute node with the pinned open checkpoint, run:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.qasper.section_bounds DATA_DIRECTORY CHECKPOINT NEW_OUTPUT_DIRECTORY
```

The release includes the replay and aggregate metrics. The upstream corpus and individual rows stay outside the source archive.
