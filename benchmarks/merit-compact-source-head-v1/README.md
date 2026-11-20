# Compare two compact source checks with person-level suitability

Use the [MERIT assessor release](https://aclanthology.org/2026.emnlp-main.2227/)
and the frozen connected-component split from the
[lightweight MERIT benchmark](../merit-lightweight-decision-v1).
Groups sharing a target title or exact normalized reviewer
publication-title multiset are kept together. The split holds
356 fit, 125 calibration and 129 test pairs. The test includes
45 positive pairs. Positive means a reviewer self-assessment
rating of four or five. These are person-level suitability
labels for supplied paper–profile pairs. The data neither
annotate evidentiary spans nor give an exhaustive ranking
of all available experts. The public release has already
appeared in earlier studies.

## Fixed source budget and decision baselines

- **Best dense source / dense head:** repeat the existing
  embedding score and the dense-and-lexical suitability head.
- **Compact maximum:** select the two highest-scoring owned
  publications from the same embedding, then read both with
  one batched Qwen3-Reranker-0.6B forward. Use the maximum of
  the two target-paper technical-relevance scores.
- **Compact head:** train a regularized logistic head that
  adds the two compact margins, their gap and lexical
  complementarity to the existing dense-and-lexical features.
  Fit on the dense-selected two-paper view.
- **Complement head:** keep the first paper and replace
  the second by the publication covering the most remaining
  target terms. Keep the same compact model and weights.
- **4B readers:** repeat the first-two and full-history
  prompted Qwen3-4B-Instruct-2507 baselines. These are generic
  open model controls. They are distinct from the published
  reviewer-tuned MERIT assessor.

The compact verifier reads title and abstract on both sides.
The 4B controls also see the target introduction. Fit all
heads on the original fit partition. Compare a balanced
accuracy cutoff with a second cutoff that accepts at least
90% of positive calibration pairs. These empirical cutoffs
are frozen before the test and do not guarantee held recall.

The test makes a separate live model call for each arm.
Rotate the order across cases and reverse the initial order
on repeat. Warm P95 includes query encoding, source selection,
tokenization and inference when that arm uses them. Profile
embeddings are resident. Build time, retrieval over the full
catalog, networking, and source acquisition are outside
the measurement.

## Results

Both tasks of Slurm array `45085` on the `ws3` RTX 5090
reproduce the fitted thresholds and all classification counts.
The historical dense and generic-reader baselines reproduce
their original balanced-cutoff decisions in aggregate.
No 0.6B input is truncated.

| Route at its balanced cutoff | True positives / 45 | False positives / 84 | Balanced accuracy | Positive F1 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Best dense source | 12 | 4 | 60.95% | 39.34% | 8.79–8.99 ms |
| Dense head | 18 | 7 | 65.83% | 51.43% | 9.04–9.10 ms |
| Compact maximum | 31 | 19 | 73.13% | 65.26% | 30.47–30.56 ms |
| Compact head | 20 | 6 | 68.65% | 56.34% | 30.23–31.57 ms |
| Complement head | 20 | 6 | 68.65% | 56.34% | 30.22–32.20 ms |
| Generic 4B, first two | 27 | 31 | 61.55% | 52.43% | 142.18–147.65 ms |
| Generic 4B, full history | 31 | 25 | 69.56% | 61.39% | 209.73–211.22 ms |

The compact maximum matches the full generic reader's true
positive count and yields six fewer false positives. Its P95
is about 85% lower. The same true positive count does not
imply agreement on which reviewer profiles pass. Its paired
accuracy difference against the dense head is +0.78 percentage
points with a descriptive component-bootstrap interval from
−9.33 to +7.87 points. Compared with the compact maximum,
the fitted head trades true positives for fewer false positives.
Lexical complementarity does not improve the tested balanced
operating point.

At the 90%-retention calibration threshold, the best dense
source retains 36 held positives with 27 false positives.
The compact maximum retains 35 and admits 29, with a higher
P95. The preferred component therefore depends on the recall
operating point. Current counts do not support treating source
relevance as a reliable hard expert filter. The scope is a
reused assessor cohort and generic open baselines.
[Aggregates for both operating points](results.json).

## Reproduce

Use the pinned public test release and verified local open
model checkpoints on a Slurm GPU. Repeat with
`--order-offset 1` and a fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.decision.merit_compact_source_head \
  MERIT_ASSESSOR_JSON QWEN_EMBEDDING QWEN_RERANKER QWEN_READER \
  benchmarks/merit-lightweight-decision-v1/results.json NEW_RESULTS
```
