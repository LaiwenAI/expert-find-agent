# Route high-cost verification on the same passage

Use the [FollowIR](https://aclanthology.org/2025.naacl-long.597/)
human assessments and the original
[paired-criterion cohort](../followir-paired-criterion-gate-v1).
Core17 and News21 provide 619 development rows. The Robust04
evaluation covers 575 rows from 50 topics, including 194 stable
positives, 181 documents made nonrelevant by the revised
criterion, and 200 stable negatives. Robust04 has appeared in
earlier studies and is held apart from fitting in this comparison.
These are document labels. The inspected passage may not contain
all of the evidence used by the original assessor.

Keep the same topic, revised criterion and first 1,800 document
characters for the Qwen3 0.6B reranker and the Qwen3 4B instruction
reader. Both score a binary decision. The model and its prompt
change together. Calibrate each model independently, taking the
largest cutoff needed to accept at most 10% of each negative
stratum in each development collection. Count false acceptances
on revised-criterion violations separately from stable negatives.
Include zero cutoffs as a diagnostic.

## Fixed-budget replay

Route at most the floor of a quarter or half of each topic's
documents. Compare a fixed hash, proximity to the 0.6B acceptance
cutoff, and prioritizing 0.6B rejects or accepts before proximity.
Compare a ridge predictor fitted on development using the 0.6B
margin, response side, lexical overlap with query and criterion,
and document length. The target is the utility gain from replacing
the 0.6B decision with 4B. A correct decision on an invalidated
document counts twice. Correct decisions on all other documents
count once. The best possible replacement decisions given these
two readers and the gold labels provide a hindsight bound.
Inference priorities do not use 4B scores or test labels.

Repeated GPU scoring in Slurm array `44954` reversed the collection
order and reproduced every input margin. Neither model clipped
an input. The compact baseline reproduced the previously
published decisions at both cutoffs. At zero cutoffs, 0.6B
accepts 39 stable positives and 19 invalidated documents. The 4B
reader accepts 76 stable positives and 51 invalidated documents.

The table shows the transferred cutoffs, 3.375 for 0.6B and 23.0
for 4B. The negative caps were fitted on development. The main
table reports the quarter-budget operating point.

| Route | 4B evaluations | Stable positives accepted | Invalidated accepted | Stable negatives accepted |
| --- | ---: | ---: | ---: | ---: |
| All 0.6B | 0 | 5 | 3 | 0 |
| All 4B | 575 | 20 | 9 | 0 |
| Hash priority | 140 | 9 | 4 | 0 |
| Near 0.6B cutoff | 140 | 15 | 5 | 0 |
| Reject first | 140 | 16 | 8 | 0 |
| Accept first | 140 | 14 | 5 | 0 |
| Ridge gain priority | 140 | 5 | 0 | 0 |
| Hindsight bound | 20 | 22 | 0 | 0 |

At half budget (285 calls), boundary routing accepts 19 positives
and nine invalidated documents. The hash arm accepts 17 and five.
The ranking of priorities depends on whether the endpoint rewards
coverage or protection against false acceptance. The quarter
budget accepts 15 stable positives versus 20 with all 4B, with
four fewer criterion violations. The descriptive topic-bootstrap
interval for the quarter route’s weighted gain over 0.6B alone
is −0.14 to +0.40 per topic. The frozen ridge gain predictor
reduces violations and forfeits the positive recovery. Hindsight
has substantial remaining room, but uses the held-out labels.
[Aggregate results](results.json).

## Live verification

Both tasks of Slurm array `44960` ran on one RTX 5090. Each
topic sends only its revised criterion. The quarter route scores
every passage once with 0.6B, then invokes 4B only on the
selected subset. Interleave the three arms and reverse the
topic order for the repeat.

| Route | Stable positives | Invalidated accepted | 4B evaluations | Topic verification P95 |
| --- | ---: | ---: | ---: | ---: |
| All 0.6B | 5 | 3 | 0 | 83.36–84.19 ms |
| All 4B | 20 | 9 | 575 | 414.76–415.65 ms |
| Quarter, near cutoff | 16 | 4 | 140 | 185.30–185.31 ms |

The quarter route reduces P95 by about 55% compared with
always using 4B. The two batch layouts produce different selected
documents on one topic and change two verification decisions
from the frozen score replay. Both live runs give the same
decisions in aggregate. The quarter route accepts four fewer
stable positives than all 4B and five fewer criterion violations.
This is a verified component trade-off. The timer starts with
resident passages and covers tokenization, the actual forwards
and routing. It excludes retrieval, source I/O, startup, queueing
and transport. The result does not establish quality
noninferiority or expert ranking effectiveness.

## Reproduce

Run the score and live modules in a Slurm GPU allocation with
the pinned FollowIR projections and local frozen checkpoints.
Repeat each with `--order-offset 1`, then compare the outputs.
Keep only aggregate results in the release.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m experiments.followir.same_passage_routing \
  CORE17 NEWS21 ROBUST04 SMALL_CHECKPOINT LARGE_CHECKPOINT SCORES_0
python -m experiments.followir.same_passage_routing \
  CORE17 NEWS21 ROBUST04 SMALL_CHECKPOINT LARGE_CHECKPOINT SCORES_1 --order-offset 1
python -m experiments.followir.evaluate_same_passage_routing \
  SCORES_0 SCORES_1 benchmarks/followir-paired-criterion-gate-v1/results.json NEW_AGGREGATE
python -m experiments.followir.live_same_passage_routing \
  CORE17 NEWS21 ROBUST04 SMALL_CHECKPOINT LARGE_CHECKPOINT SCORES_0 LIVE_0
```
