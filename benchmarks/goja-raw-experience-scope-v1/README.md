# Extract experience links from raw job advertisements

Extend the [annotated-anchor candidate study](../goja-scope-nomination-v1)
to the full anonymized [GOJA release](https://github.com/KruegerETRF/GOJA/tree/06c9887cceafdacec319ea222b16a4419870eca6).
Predict experience anchors and their attached targets from raw ad
text. The target labels are exactly the experience-to-process and
experience-to-work-content edges defined in the
[original binding experiment](../goja-experience-binding-v1).
Retain every eligible edge and ads with zero such edges. This
experiment concerns relation extraction in German job ads.

## Fixed protocol

Hash normalized ad text and use buckets 0–3 to fit, 4–5 to choose
operating points, and 6–9 to test. The splits contain 199, 115 and
186 ads respectively. The test contains 318 linked experience
anchors and 561 exact edges. It includes 56 ads without an annotated
edge of the eligible types. Ads may share templates even after
exact-text deduplication.

1. Propose one- to six-word anchors ending at word boundaries.
   Collect spans containing an experience cue or matching a surface
   form seen in training. The gazetteer keeps only seen forms and
   prefers longer nonoverlapping matches. The learned model scores
   layout, token cues and regularized lexical features, keeps
   nonoverlapping spans in score order, and chooses its cutoff by
   exact anchor F1 on the development ads.
2. Apply the trained [scope-span proposal and ranking](../goja-scope-nomination-v1)
   around each returned anchor. Keep the twelve highest-ranked
   spans. The training-time vocabularies and lexical statistics
   omit the current training ad. Inference uses the fixed
   training bank and observes no test labels.
3. Compare issuing one scope per anchor, the strongest fixed count
   from one to five, and issuing each candidate above a single
   calibrated score. Choose the fixed count and score cutoff by
   exact pair F1 on the development ads. This is a follow-up on
   the same held-out ads used for the one-scope diagnosis.
   The oracle supplies annotated anchors and uses the same binding
   ranker as a stage diagnostic.

## Exact results

| Route | Issued links | Correct links | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gazetteer · top one | 650 | 103 | 15.8% | 18.4% | 17.0% |
| Learned anchors · top one | 263 | 82 | 31.2% | 14.6% | 19.9% |
| Learned anchors · fixed top three | 789 | 156 | 19.8% | 27.8% | 23.1% |
| Learned anchors · calibrated cutoff | 767 | 145 | 18.9% | 25.8% | 21.8% |
| Annotated anchors · top one | 318 | 183 | 57.5% | 32.6% | 41.6% |

The anchor menu covers 224 of 318 linked gold anchors. The
gazetteer issues 650 anchors, including 188 exact hits, for an
anchor F1 of 38.8%. The learned model issues 263 anchors, with
161 hits and F1 of 55.4%. The learned anchors preserve 340 of
561 eligible edges. Among those, 311 have a generated target,
269 are in the first twelve, and 82 are ranked first. With
annotated anchors, all 561 edges enter the binding stage. The
generator covers 504, the shortlist preserves 456, and 183
rank first. The shortlisted gold targets attach to 291 distinct
anchors, which gives the best possible correct-edge count under
the one-target restriction on that route.

The development split selects three as the best fixed count.
It has the highest observed test F1 among the raw-text routes.
The calibrated cutoff issues a link on eight of the 56 negative
ads, versus thirteen for the fixed learned route and thirty-three
for the gazetteer. Those counts refer to the eligible annotation
edges. This pilot leaves substantial room in both anchor coverage
and target ranking. It does not establish extraction of mandatory
qualifications, evidence attribution to an expert, or end-to-end
expert recommendation quality.

Slurm job 46744 reproduces all scores and the outcome signature
in two traversal orders. [Aggregate results](results.json)

## Reproduce

Download the pinned anonymized ZIP from the
[original benchmark](../goja-experience-binding-v1#reproduce).
Run the following on a Slurm compute allocation. Repeat with
`--reverse` to verify traversal stability.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.retrieval.goja_raw_experience_scope GOJA_ZIP NEW_RESULTS
```
