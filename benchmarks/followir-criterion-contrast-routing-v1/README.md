# Separate topic and criterion signals for selective review

Use the exact FollowIR projections and scored passages in the
[same-passage verifier study](../followir-same-passage-routing-v1).
The 619 Core17 and News21 rows supply calibration and fitting.
The previously explored Robust04 panel has 575 rows and 50 topics.
It contains 194 stable positives, 181 topic documents invalidated
by a changed criterion, and 200 stable negatives.

Score the current query without its criterion on the same
1,800-character passage using the same 0.6B reranker. Keep the
existing score with the current criterion, thresholds (3.375
and 23.0), 4B decisions, and per-topic budgets. The query-only
view uses no earlier relevance instruction or test labels.
Test relevance is a document label. It does not establish whether
the displayed passage contains enough supporting evidence.

## Equal-call replay

Four comparisons use the additional topic cue: rank by the
topic-only margin, by the absolute margin change, or by the
largest drop among 0.6B rejects first. Add the topic margin,
signed and absolute changes, and the change-by-acceptance
interaction to the existing ridge gain predictor with the same
regularization and objective. Keep nearest-cutoff and the
original ridge predictor as controls.

After observing the individual priorities on this reused panel,
we also tried a mixed route. Preserve all but the last
nearest-cutoff review slot on each topic. Reassign the last slot
to the candidate with the highest topic-only score among the
remaining documents or among the next four on the cutoff list.
The latter limits the additional compact forward to four
document scores per active topic. Treat these two choices as
exploratory variants.

Slurm array `44969` on an RTX 5090 reproduced every topic-only
margin with reversed collection order. The following policies
all sent 140 passages to 4B in quarter-budget replay.

| Priority | Stable positives accepted | Invalidated accepted |
| --- | ---: | ---: |
| Near cutoff | 15 | 5 |
| Topic-only margin | 13 | 4 |
| Absolute criterion change | 8 | 4 |
| Prioritize criterion-suppressed rejects | 6 | 5 |
| Ridge without contrast | 5 | 0 |
| Ridge with contrast | 5 | 0 |
| Mixed, score all remaining | 15 | 3 |
| Mixed, score next four | 15 | 3 |

All listed routes accept zero stable negatives. Ranking by
topic-only margin reviews two of the six stable positives missed
by the local boundary rule, while yielding two fewer stable
positives overall. The mixed quarter-budget route finds one of
the six and changes which positives pass. At half budget (285
4B evaluations), the mixed route and boundary route both pass 19
stable positives. The mixed route passes eight invalidated
documents and the boundary route passes nine. The next-four
route uses 200 additional compact scores at either budget.
Across the 50 topics, the paired descriptive interval for
the quarter-budget change was −0.06 to +0.06 accepted positives
per topic and −0.10 to 0.00 accepted invalidated documents per
topic.
[Aggregate results](results.json).

## Live verification

Two Slurm tasks in array `44978` on an RTX 5090 reversed
the topic order with the same accepted-document totals.
The mixed route executed a new compact forward on only the
four remaining candidates closest to the cutoff, per topic.
Both selective routes executed 140 4B document evaluations.

| Route | Stable positives | Invalidated accepted | Added compact scores | P95 |
| --- | ---: | ---: | ---: | ---: |
| All 0.6B | 5 | 3 | 0 | 83.29–83.52 ms |
| All 4B | 20 | 9 | 0 | 412.38–413.61 ms |
| Quarter, near cutoff | 16 | 4 | 0 | 183.64–185.60 ms |
| Quarter, mixed next four | 15 | 3 | 200 | 208.42–210.01 ms |

Live batch shapes changed two document decisions per
selective route versus its cached-margin replay. Compared with
nearest-cutoff, the mixed route lost one positive, avoided one
criterion violation, and increased P95 by about 13%. The timing
covers tokenization, actual model calls and routing for resident
passages. Retrieval, storage, queues and transport remain outside
the measurement. The mixed rule does not improve the measured
speed–quality trade-off. A topic score already computed on the
same passage upstream can be tested separately at no added
model call.

## Reproduce

Run the scorer and live module inside a Slurm GPU allocation.
Repeat both GPU commands with `--order-offset 1` and fresh output
paths. Evaluate matched cached scores on CPU.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m experiments.followir.criterion_contrast_routing \
  CORE17 NEWS21 ROBUST04 SMALL_CHECKPOINT BASE_SCORES NEW_TOPIC_SCORES
python -m experiments.followir.evaluate_criterion_contrast_routing \
  BASE_SCORES TOPIC_SCORES_0 TOPIC_SCORES_1 \
  benchmarks/followir-same-passage-routing-v1/results.json NEW_AGGREGATE
python -m experiments.followir.live_same_passage_routing \
  CORE17 NEWS21 ROBUST04 SMALL_CHECKPOINT LARGE_CHECKPOINT BASE_SCORES NEW_LIVE_RESULT \
  --contrast TOPIC_SCORES_0
```
