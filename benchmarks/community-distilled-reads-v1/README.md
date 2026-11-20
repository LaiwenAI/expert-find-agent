# Distill which authors' answers to verify on the first query

Use the frozen community panels and the dense source-first path from
[the initial route comparison](../community-first-query-owner-v1).
The online question is encoded with Qwen3-Embedding-0.6B. Rank original
answer bodies and retain the best answer from each of the ten highest
distinct posting accounts. Neither candidate generation nor the student
sees author profiles, canonical question text, or accept labels.

Train a six-feature ridge student on Cross Validated 2024. The teacher
is the existing Qwen3-Reranker-0.6B answer verifier. Predict its clipped
yes-minus-no margin on ranks 3–10 using dense similarity, inverse rank,
unweighted and IDF-weighted query term coverage, answer term coverage,
and answer vocabulary size. Features use the inspected text window.
Train only once for each run, without citation labels. Freeze weights
before evaluating the later Cross Validated panel and MathOverflow.
Training candidate count and wall time are recorded separately.
See [RankDistil](https://proceedings.mlr.press/v130/reddi21a.html) for
prior work on top-ranked knowledge distillation. This experiment tests
a small pointwise control for verification allocation.

## Controls

- **Dense3, Dense5, Dense10** verify the first 3, 5, or 10 distinct
  authors in dense order. Dense5 must reproduce the previous published
  selection and issuance counts. Dense10 diagnoses additional
  reachable evidence under a larger budget.
- **Overlap3** preserves the first two and uses the final read on the
  answer with the largest IDF query overlap among ranks 3–10.
- **Student3 and Student5** also preserve the first two, then assign
  the remaining verification reads by the student's predicted margin.

All arms use the same 950-character query and 850-character answer
caps in a single live verifier batch. Only the verifier can issue a
citation. Sort positive margins, retaining at most five results.
Record answer checks, selected known answers and issued known answers.
For the matched-issuance diagnostic, truncate both outputs to the
smaller number of positive results on each query. This diagnostic is
computed after inference. Unlisted citations remain unjudged.
Supported P@5 uses a fixed denominator of five and a paired bootstrap
by shared labeled answer. P95 includes fresh query encoding, candidate
selection and verification, with one warm local client. Run twice
while rotating evaluation order.

## Results

Both tasks of Slurm array `45673` completed on a ws3 RTX 5090.
They reproduced the earlier five-read baseline, learned weights,
and all non-timing outcomes. Teacher labeling and fitting took
approximately 4.8 seconds per run, excluding index construction.

| Arm | CV 2025 known / issued | MathOverflow known / issued | Warm CV P95 | Warm MathOverflow P95 |
| --- | ---: | ---: | ---: | ---: |
| Dense3 | 16 / 23 | 13 / 13 | 33.46–33.58 ms | 34.02–34.12 ms |
| Overlap3 | 15 / 24 | 13 / 13 | 34.22–34.42 ms | 31.94–32.38 ms |
| Student3 | 16 / 25 | 13 / 13 | 34.17–34.22 ms | 32.09–32.19 ms |
| Dense5 | 16 / 26 | 13 / 13 | 47.03–47.43 ms | 47.62–47.68 ms |
| Student5 | 16 / 28 | 13 / 13 | 47.54–47.60 ms | 47.59–47.71 ms |
| Dense10 | 16 / 30 | 13 / 13 | 84.24–84.25 ms | 88.15–88.29 ms |

The simple fixed three-read baseline uses 40% fewer verification
pairs and reduces the two held-panel P95s by approximately 28–29%
relative to five reads. It recovers the same number of known
citations on both panels. Across the 47 requests, going from three
to ten reads exposes four extra known answers, with no increase in
verified known returns. The CV 2025 panel issues three fewer
citations with three reads than with five, with the same known
count. Three- and five-read distillation add no known returns
against the corresponding dense baselines, including with output
size matched per request.

Keep Dense3 alongside Dense5 for future tests. The small, partially
judged panels do not establish noninferiority or production latency.
The larger-budget result motivates testing retrieval and the reader
separately on independently judged requests where relevant evidence
is not concentrated in the first three candidates.

[Aggregate results and timings for both runs](results.json).

## Reproduce

Restore the prior three public snapshots and pinned model checkpoints.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.community.distilled_first_query_reads \
  CV_2024 CV_2025 MATHOVERFLOW QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/community-first-query-owner-v1/results.json NEW_RESULTS \
  --order-offset 0
```
