# Separate retrieval order from verification and issuance

A verifier can change candidate rank, or decide whether to issue a
candidate while preserving retrieval order. Compare these policies
with the same nominated documents, excerpts, and reader scores.

- **Reorder:** choose the highest-margin checked candidate and
  issue it above the cutoff.
- **First passing:** issue the first checked candidate in retrieval
  order with a margin above the cutoff. Later candidates do not
  outrank an earlier candidate that passed.
- **Top only:** issue the original leading candidate if it passes.
  The measured one-candidate route skips all other checks.
- **Retrieve:** accept or abstain using the leading candidate's
  retrieval confidence, with no reader call.

## CrossPage

Reuse the [110 competing candidate pools](../crosspage-competing-pools-v1).
Use either whole-query dense or condition-wise hybrid document order.
For each checked document, nominate one distinct page per condition
with the hybrid page scorer and read the same page prefixes. The
document margin is the minimum of the condition margins.
Evaluate reorder and first-passing from identical live scores. The
reader is batched, so both policies have the same observed read cost.
Thresholds of 0 and −4 are diagnostics. Also exhaustively sweep
cutoffs on the observed set at the same caps of five or ten partial
documents issued. Tied scores cross the threshold together. These
retrospective caps do not calibrate a deployment threshold.

| Candidate order and policy | Page checks | P95 (ms) | Complete at ≤5 partial | Complete at ≤10 partial |
| --- | ---: | ---: | ---: | ---: |
| Whole dense, retrieve | 0 | 8.09–8.31 | 9 | 18 |
| Condition BM25, retrieve | 0 | 0.16–0.17 | 9 | 16 |
| Whole dense, top only | 225 | 32.80–33.25 | 6 | 10 |
| Whole dense, first passing, two candidates | 450 | 59.18–60.04 | 8 | 8 |
| Whole dense, reorder, two candidates | 450 | 59.18–60.04 | 8 | 8 |
| Whole dense, first passing, four candidates | 900 | 113.93–114.74 | 10 | 10 |
| Whole dense, reorder, four candidates | 900 | 113.93–114.74 | 9 | 9 |
| Hybrid, first passing, two candidates | 450 | 43.71–44.20 | 8 | 8 |
| Hybrid, reorder, two candidates | 450 | 43.71–44.20 | 8 | 8 |

Preserving order helps slightly in the four-candidate whole-dense
comparison. At the cap of ten partial outputs, the cheaper whole-query
dense route retains more complete document labels than any reader
route. This benchmark does not grade the exact evidence excerpt.

## RELISH

Reuse the [source-disjoint RELISH panel](../relish-risk-routing-v1).
Each request supplies 24 judged candidates. Nominate by the original
whole-abstract vector and give both verification policies the same
title and first three abstract sentences for each checked candidate.
The no-reader route retains the first whole-abstract candidate and
uses its compact-view cosine as its issuance score.

For each policy, maximize grade-2 issuance on 55 calibration requests
while issuing at most five grade-0 or grade-1 papers. Break ties in
favor of fewer adverse results, then the lower threshold. Freeze
thresholds before evaluating the 100 held requests. Issue at most one
paper per request. A grade is scientific relevance to the seed article.

| Policy | Checked per request | Strong / 100 | Partial or nonrelevant / 100 | P95 (ms) | Paired strong difference vs retrieve, 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retrieve | 0 | 39 | 8 | 8.80–8.86 | Reference |
| Top only | 1 | 46 | 12 | 17.39–17.61 | +7 pp [−2, +16] |
| First passing | 2 | 41 | 7 | 21.00–21.07 | +2 pp [−8, +12] |
| Reorder | 2 | 42 | 8 | 21.00–21.07 | +3 pp [−7, +13] |
| First passing | 4 | 30 | 5 | 31.27–31.51 | −9 pp [−19, +1] |
| Reorder | 4 | 50 | 12 | 31.27–31.51 | +11 pp [+2, +21] |

Reordering four candidates gains strong results alongside more
partial or nonrelevant ones, and violates the calibration risk target
on the held split. Two-candidate policies stay within that target, but
their differences from cheap retrieval are uncertain. RELISH does
not label condition-specific excerpts or personal contribution.

## Reproduce

Run on a Slurm CUDA allocation with the pinned models. Run each
command for offsets 0 and 1. Both benchmarks time initialized
serial requests through encoding, candidate selection, verification
when enabled, and serialization. The CrossPage and RELISH Slurm arrays
45826 and 45831 reproduce all non-timing fields across rotated executions
on ws3 RTX 5090. See [CrossPage results](crosspage-results.json) and
[RELISH results](relish-results.json).

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.crosspage_certificate_ablation \
  CROSSPAGE_PANEL CROSSPAGE_MANIFEST QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS --offset 0

PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.csfcube.relish_certificate_ablation \
  RELISH_PANEL RELISH_MANIFEST QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS --offset 0
```
