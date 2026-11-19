# Budgeted source expansion for expert attribution

Reuse the [frozen transfer catalog and queries](../plos-body-transfer-v1/README.md). Rank titles and abstract prefixes with the exact same dense and lexical reciprocal-rank fusion. Retain the first four nominations from that pass. Keep every first source that the previous dense-then-body policy accepts. The new actions run only when that policy abstains.

| Arm | Action on abstention |
| --- | --- |
| Top one | Return no source. |
| Reused cutoff, depth four | Check ranks two through four in order. Stop at the first publication whose body passes the previous top-one lexical cutoff. |
| Calibrated cutoff, depths two and four | Stop at the first rank that passes the same expansion cutoff. |
| Flat depth four | Scan the bodies through rank four and choose the highest score if it passes the expansion cutoff. |
| Relative dense score, depth four | Check the score minus the median of the dense top four. Pick the first subsequent nomination beyond the relative cutoff, without an extra body scan. |

For each clue form, fit the expansion cutoffs on the old development publications. Consider all incorrect proposals at ranks two through four in the present and withheld views after first-source abstention. The strict cutoff is the maximum such score. It is estimated separately for lexical coverage and the relative dense score. The latter is a simple score-distribution comparator inspired by [The Magnitude Mirage](https://aclanthology.org/2026.emnlp-main.517/).

Keep the same author verifier for all selected sources. Count correct nominations and how many missed targets could be reached at each depth. Measure the number of inspected candidate bodies and verification calls, as well as factored P95 over retrieval, inspected bodies and author verification. A body read is one source-local lexical scan over in-memory windows. Every arm stops with at most one source. A clean success requires a nonempty recommendation with every returned author supported.

This is a diagnostic replay on previously used transfer topics, with the expansion policies calibrated on the older development topics. Paired intervals resample connected author-name groups. Source-withholding counts wrong recommendations relative to an exact-study requirement. Timing excludes corpus preparation, network service and concurrent queues.

## RTX 5090 results

Slurm job 42677 on ws5 used 197 catalog papers, 30 old development queries and 72 reused test queries in 70 author-name clusters. The top-one route reproduced the pinned transfer counts for source match, supported and clean recommendations, wrong-source recommendations and author model calls.

| Clue | Route | Correct source | Any supported author | Clean success | Wrong-source recommendation, present / withheld | Body scans, present / withheld |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Excerpt | Top one | 64 | 57 | 37 | 0 / 0 | 20 / 72 |
| Excerpt | Reused cutoff, depth four | 65 | 58 | 37 | 1 / 1 | 42 / 287 |
| Excerpt | Calibrated depth two | 65 | 58 | 37 | 0 / 0 | 28 / 144 |
| Excerpt | Calibrated depth four | 65 | 58 | 37 | 1 / 1 | 42 / 287 |
| Excerpt | Flat depth four | 65 | 58 | 37 | 1 / 1 | 44 / 288 |
| Excerpt | Relative dense score | 64 | 57 | 37 | 0 / 0 | 20 / 72 |
| Keywords | Top one | 50 | 42 | 27 | 1 / 4 | 31 / 68 |
| Keywords | Reused cutoff, depth four | 53 | 45 | 27 | 1 / 4 | 89 / 272 |
| Keywords | Calibrated depth two | 52 | 44 | 27 | 1 / 4 | 52 / 136 |
| Keywords | Calibrated depth four | 53 | 45 | 27 | 1 / 4 | 89 / 272 |
| Keywords | Flat depth four | 53 | 45 | 27 | 1 / 4 | 94 / 272 |
| Keywords | Relative dense score | 50 | 42 | 27 | 1 / 4 | 31 / 68 |

With the source present, the intended paper is in the first, first two and first four positions on 70, 72 and 72 excerpt queries, versus 57, 61 and 65 keyword queries. After first-source abstention only two excerpt targets and seven keyword targets are reachable at positions two through four. One keyword target is blocked by an already accepted wrong source. The remaining seven keyword targets fall below rank four.

The calibrated depth-four cascade adds one supported excerpt query and three supported keyword queries. Each newly supported query also returns a wrong author on the correct source, so clean success is unchanged. It also introduces one wrong-source recommendation in each excerpt view. The flat comparator selects the same sources at higher body-read counts. The relative score adds no source acceptance at this calibration point. Calibrated depth two avoids the new wrong-source recommendation in this sample and still adds no clean success.

Present-view factored P95 is 99.03 ms for excerpts and 99.06 ms for keywords across all tested arms. With sources withheld, depth four raises the excerpt P95 from 8.01 to 8.68 ms and the keyword P95 from 7.40 to 7.47 ms. The read counts expose extra work even when author-verifier calls dominate the present-view tail. Keep the top-one fallback as the reference. Source recall and author verification are separate bottlenecks in this task. [Aggregate results](results.json)

## Reproduce

Restore the three publisher captures following the [transfer benchmark](../plos-body-transfer-v1/README.md) and use the same open checkpoints in a Slurm GPU job.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.legacy_source_expansion \
    OLD_SNAPSHOT FIRST_TRANSFER FRESH_TRANSFER QWEN_EMBEDDING QWEN4B \
    OPEN_ATTRIBUTION_RESULTS BODY_CONFIRMATION_RESULTS CASCADE_RESULTS \
    BODY_TRANSFER_RESULTS NEW_RESULTS
```
