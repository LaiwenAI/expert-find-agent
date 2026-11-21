# Test expected support on real community questions

Transfer the source-priority idea from the [TREC source ranking comparison](../trec-source-owner-contrast-v1)
to the public [community witness panels](../community-joint-answer-witness-v1).
The test uses three archived panels of Cross Validated and MathOverflow
questions and accepted answers. Each paired request contains two
questions with distinct known answers written by at least one common
author. The public accepted-answer flag represents the asker's
selection. The original poster's assessment does not fully judge
other answers. [Acceptance policy](https://stackoverflow.com/help/accepted-answer).

Replay the published topic-only, staged repair and joint
same-author plans alongside the new routes. Within the replay, use identical
depth-64 answer-body ranks, answer excerpts and Qwen checkpoints.
All plans read three distinct answers and perform six logical
verifier checks, one per question and opened answer.

- **Additive** prioritizes each answer by the sum of its reciprocal
  ranks for the two questions. This is the cross-task analogue of
  aggregating each source's expected support. With one owner per
  answer, the two terms refer to distinct conditions for that owner.
- **Pair static** chooses the highest-product same-author pair, then
  adds the answer that maximizes estimated probability of recovering
  a distinct same-author pair. Reciprocal ranks serve as fixed
  heuristics for the binary support events. The rule calculates the
  union probability under independent support events, accounting for
  shared reads across possible pairs.
- **Pair adaptive** takes the same seed pair. It sets the four observed
  seed support events to zero or one and chooses the final answer
  using the same probability calculation. The action reads no
  verifier labels outside the seed.

The model's verdict influences which result is issued. Exact known
recovery requires both labeled answers from that same author. Report
availability after source selection, actual recovery, recovery on a
previously fixed disjoint-question subset, and recovery matched to
staged repair's issuance. Matching retains each policy's most
confident outputs up to the smaller cohort count. It uses no gold
labels, can retain different requests, and is a retrospective
diagnostic. These snapshots were already used in the published
community baseline. They are independent of TREC, with no tuning
against community target labels for these three new rules.

## Results

All policies use three reads and six logical checks per request.
Each cell gives known pairs available in the planned documents
and known pairs returned after verification.

| Panel | Requests | Staged | Joint | Additive | Pair static | Pair adaptive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross Validated 2024 | 136 | 62 / 17 | 80 / 18 | 52 / 12 | 79 / 19 | 83 / 20 |
| Cross Validated 2025 | 37 | 18 / 7 | 22 / 5 | 11 / 2 | 28 / 5 | 26 / 7 |
| MathOverflow | 143 | 104 / 66 | 109 / 65 | 66 / 43 | 122 / 70 | 121 / 70 |

Both pair planners expose more known pairs than staged repair in all
three panels. On MathOverflow, static planning returns four extra
known pairs and issues four extra results. No method improves
known recovery when matched to staged repair's issuance on any
panel. On the disjoint-question subset, static planning yields
2, 1 and 30 returns compared with 2, 2 and 30 for staged repair.
The adaptive variant yields 2, 2 and 30. All issued pairs in these
selected positive panels match a known pair. Unselected answers and
general query precision remain unjudged. The gain in availability
supports evaluating the static pair plan as a parallel-read option.
Determine its serving benefit with actual latency, and test retrieval
and verified precision on independently annotated traffic before
claiming an overall gain.

Slurm tasks 46882_0 and 46882_1 ran on ws1 (RTX 4090, PyTorch
2.15.0+cu132, Transformers 5.23.0). They reproduce the results and
outcome signatures with opposite panel orders. The old RTX 5090
aggregate differs on some baseline counts under this setup, most
clearly staged Cross Validated 2024 (24 previously, 17 here).
The new comparisons use contemporaneous controls on the same
runtime. The results preserve the differences against the previous
aggregate and the input hashes. [Aggregate results](results.json).

## Reproduce

Restore the pinned community snapshots, checkpoints and archived
baseline scorecard. Run in a Slurm allocation with a GPU and a fresh
results path.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.community.expected_support_dispatch \
  STATS_2024 STATS_2025 MATHOVERFLOW QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/community-joint-answer-witness-v1/results.json NEW_RESULTS
```

Repeat with `--reverse` and a different output path.
