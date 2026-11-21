# Nominate experience scope spans directly from job-ad text

Reuse the frozen [150-anchor GOJA test](../goja-experience-binding-v1)
on 65 anonymized German job ads. The earlier binding experiment
gave each method annotated candidate spans. This comparison gives
candidate generation only raw ad text and the selected experience
anchor's offsets. The test remains conditional on an annotated
anchor and a target in the earlier gold menu. All target coordinates
and links are kept on the evaluation side.

## Candidate generation and ranking

- **Word spans.** Enumerate disjoint one- to five-word spans within
  180 characters of the anchor. Keep a token's internal hyphen or
  slash and stop a span at punctuation or a line break.
- **Compound boundaries.** Keep all word spans and allow the longest
  known proper prefix and proper suffix of each token. Learn the
  vocabulary from single-token target phrases in training ads only,
  with at least three characters. When training the ranker, omit the
  current ad from the boundary vocabulary and lexical statistics.
- **Ranking.** Compare character gap, same line then character gap,
  a regularized structural ranker and the same ranker with lexical
  counts. The lexical features use prefix and suffix words,
  three-character word suffixes, and the whole proposed phrase.
  Every method receives the same top-one, top-five and top-twelve
  candidate budgets within its generator.

An exact hit requires both original offsets. The overlap diagnostic
requires at least 0.5 character intersection over union and no
overlap with any other annotated alternative in the earlier menu.
Candidate ceilings count anchors with a match anywhere in the
generated set before ranking. Resample whole ads in the paired
intervals. This is an annotation-conditioned component experiment.
It does not test experience-anchor detection, hard-qualification
necessity, expert retrieval, or full service latency.

## Results

| Generator and ranker | Generated spans | Exact ceiling | Exact @1 | Exact @5 | Exact @12 | Overlap @12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Word spans, same line first | 17,217 | 97 | 10 | 29 | 74 | 101 |
| Word spans, trained lexical | 17,217 | 97 | 64 | 87 | 94 | 110 |
| Compound boundaries, same line first | 17,548 | 127 | 26 | 48 | 100 | 122 |
| Compound boundaries, trained lexical | 17,548 | 127 | 82 | 112 | 124 | 131 |

The compound expansion adds 1.9% more pre-ranking spans and
recovers 30 exact targets before ranking. For trained lexical
ranking at 12 candidates, exact coverage rises by 20.0 percentage
points (grouped 95% interval +13.7 to +26.8). At one candidate
it rises by 12.0 points (+6.0 to +18.2). The final generator still
omits 23 gold spans, and the trained ranking leaves three of its
127 covered targets outside the top 12. These outcomes separate
the candidate generation and ranking errors. A high score on supplied
mentions can coexist with a large upstream candidate gap.

Slurm job 46733 reproduces the aggregate results and per-anchor
outcome signature in both traversal orders. The source archive
remains available from [GOJA](https://github.com/KruegerETRF/GOJA/tree/06c9887cceafdacec319ea222b16a4419870eca6).
The staged results contain no advertisement text or offsets.

## Reproduce

Obtain the pinned anonymized GOJA ZIP from the
[preceding benchmark](../goja-experience-binding-v1#reproduce).
Use a Slurm CPU allocation and run again with `--reverse`.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.retrieval.goja_scope_nomination GOJA_ZIP NEW_RESULTS
```

[Complete aggregate results](results.json)
