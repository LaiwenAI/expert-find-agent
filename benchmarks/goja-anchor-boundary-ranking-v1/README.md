# Rank competing GOJA boundaries at a fixed per-ad budget

Test whether explicit boundary comparison helps select the newly
proposed spans in the [GOJA anchor expansion](../goja-anchor-candidate-expansion-v1).
Reuse the same 199 fit, 115 development and 186 test ads. Hold the
generated compound spans, target ranker and target fanout fixed.
Each route sends the same number of anchors per ad as the frozen
standard route, then issues one scope per selected anchor.
The held-out pool has 318 linked anchors and 561 annotated edges.

## Interventions

- **Boundary contrast.** Add a pairwise squared loss that asks
  the exact anchor to outrank overlapping non-gold variants.
  Use up to four competitors per exact anchor, ordered by
  intersection over union and source order for ties.
- **Nearby control.** Use the same exact positives, sampled pair
  counts, and weight. Choose non-overlapping rivals by distance
  and similar span length. Limit both arms to positives with
  both rival types available.
- **Boundary features.** Add six features to the pointwise ranker.
  Mark whether either endpoint cuts an alphanumeric token, whether
  a linking word touches either side across whitespace, whether
  a numeric duration precedes the span, and whether following
  punctuation marks its end. Compare features alone with features
  and the overlap contrast.

The pointwise objective and regularization are shared, as are
lexical features based on fit ads. Where applicable, the contrast
has a total weight of 0.25 per participating fit ad. The two
feature routes keep exactly the same candidate offsets as the
other compound routes. The comparison is exploratory on the
same held-out ads as the previous candidate study. Intervals
are descriptive and resample whole ads.

## Results

| Ranker | Exact anchors / 263 sent | Gold edges in top 12 targets | Correct top-one links |
| --- | ---: | ---: | ---: |
| Standard word-boundary reference | 161 | 269 | 82 |
| Compound reference | 155 | 259 | 79 |
| Compound with nearby control | 155 | 259 | 79 |
| Compound with overlapping contrast | 157 | 265 | 81 |
| Compound with boundary features | 162 | 268 | 85 |
| Compound with boundary features and contrast | 161 | 267 | 84 |

The overlap contrast recovers two correct links relative to the
compound reference and the matched nearby control. Explicit
features recover six relative to the compound reference, a
+1.07 percentage-point recall difference (95% interval +0.33
to +2.06). Relative to the strongest standard reference, the
gain is three links and +0.53 points (−0.60 to +1.74). The
contrast adds no observed gain once the features are available.
The experiment therefore identifies a low-cost feature change
with measurable gains within the expanded candidate route. It
does not establish a quality improvement over the standard
route or an improvement in end-to-end expert search.

Slurm job 46780 on ws3 reproduces all aggregates and outcome
signatures in both traversal orders. Standard and compound
reference arms reproduce the earlier archive. Only the
aggregate metrics and public code are included.
[Complete results](results.json)

## Reproduce

Obtain the anonymized GOJA ZIP and verify its digest using the
[original binding study](../goja-experience-binding-v1#reproduce).
Use a Slurm compute allocation and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.goja_anchor_boundary_ranking GOJA_ZIP NEW_RESULTS
```
