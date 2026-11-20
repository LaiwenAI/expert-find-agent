# Bound same-person evidence nomination by topic rank

In the [earlier comparison](../plos-topic-role-holdout-v1), selecting four
sources with a same-person role index increased both grounded and off-label
citations. Test two explanations: role postings can pull sources from deep in
the topic ranking, and the original topical acceptance cutoff may be too
lenient for the sources introduced by role screening.

Reuse the pinned PLOS pilot and the 78-article, author-disjoint replay catalog
(46 requests in 23 topic clusters). The replay catalog was evaluated in the
earlier comparison. This is a controlled follow-up on that catalog.
Use the same hybrid topic ranking, Qwen3-Reranker-0.6B, publisher role
bindings and final five-slot output. Try four-source nomination with
same-person postings or article-wide role unions, scanning the whole topic
ranking or restricting the scan to its first eight sources. Missing index
entries remain eligible. All routes confirm roles for each individual after
the topical check. The union comparison isolates the effect of using
same-person bindings during nomination. Topic-only four and eight are the
late-binding controls.

For each role route, select one cutoff on the pilot from the fixed grid
(`-2.0625, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2`). Require no more
off-label citations than topic-only four at either declaration mask, then
maximize grounded citations across masks. Freeze the cutoff before evaluating
the replay catalog. Publication tags are reference labels and can omit
relevant topics. An off-label citation is a signal for adjudication.

## Results

Slurm job `43530` completed on an RTX 5090. All four role routes chose
`-1.5` on the pilot. The original cutoff is `-2.0625`.

| Complete declaration coverage | Grounded / 230 | Off-label | Source reads | Local P95 |
| --- | ---: | ---: | ---: | ---: |
| Topic four | 37 | 4 | 184 | 34.60 ms |
| Topic eight | 49 | 14 | 368 | 64.43 ms |
| Whole-ranked union four | 42 | 10 | 184 | 37.93 ms |
| Whole-ranked person four | 52 | 16 | 184 | 37.75 ms |
| Top-eight union four | 41 | 8 | 167 | 34.82 ms |
| Top-eight person four | 44 | 10 | 125 | 34.80 ms |
| Top-eight person four, pilot cutoff | 36 | 8 | 125 | 34.79 ms |

At full coverage, bounding the person route retains seven of the original
15 additional grounded citations relative to topic four, with 32% fewer
reads than topic four. Its paired change in grounded P@5 is +3.04 percentage
points (descriptive topic-cluster 95% interval +0.87 to +5.65), accompanied
by +2.61 points of off-label citations (+0.43 to +5.22). The paired P95
change interval crosses zero (−3.01 to +0.55 ms). The matched bounded union
route obtains 41 grounded citations. The +1.30-point same-person increment
over that route has a lower interval endpoint of zero. At roughly half
declaration coverage, both bounded role routes obtain 40 grounded citations,
versus 37 for topic four.

The stricter cutoff reduces coverage and still exceeds the replay's off-label
count for topic four. None of the role routes pass both pre-existing
quality and off-label margins against topic eight. Local timing includes
query encoding, candidate selection and verification. Service, parsing and
queue costs are outside the measurement. The evidence supports bounded
nomination as a controlled alternative for future evaluation. Keep the
conservative fixed-width route as the existing same-output efficiency
control. A fresh catalog with explicit adjudication of off-label matches
would distinguish semantic false positives from incomplete publisher tags.

[Aggregate results and calibration grid](results.json)

## Reproduce

Rebuild the original projections using the
[holdout manifest and instructions](../plos-topic-role-holdout-v1).
Run with the original checkpoint revisions in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. python -m experiments.credit.bounded_topic_nomination \
  PILOT_PROJECTION HOLDOUT_PROJECTION benchmarks/plos-topic-role-holdout-v1/manifest.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
