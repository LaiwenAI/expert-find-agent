# Serve static evidence plans in a single verifier wave

The [earlier read-budget replay](../community-expected-dispatch-v1)
found more available known evidence pairs with a static same-author
plan. The effects on verified recovery depended on the issuance
budget. This experiment measures whether executing the static plan
in one verifier wave reduces client latency at the same logical budget.

Use the same frozen MathOverflow collection and open Qwen embedding
and reranking checkpoints. Take 54 of its 143 paired requests by a
fixed hash of their question identifiers. The client sends both
question texts through loopback HTTP. A single FIFO worker on one
GPU encodes each request anew and searches resident BM25 and dense
answer-body indexes. Every route reserves three answer slots and
uses up to six question–answer checks per request. The three dense
controls always read all three excerpts and make six checks. There
is no result cache or cross-request batching. Index creation,
model load, warmup, and external source retrieval are outside the
measurement.

| Service arm | Source plan | Verifier schedule |
| --- | --- | --- |
| Staged repair | Topical anchors followed by an answer selected after verifier feedback | Four checks then two |
| Static split | Same-author probability plan | Four checks then two |
| Static batched | Identical same-author plan | Six checks in one batch |
| Staged sparse | Staged repair, with out-of-rank anchors omitted from feedback | Check only eligible anchors, then any final answer needed by a proof |
| Static sparse | Same static plan, using known authorship and rank eligibility | One batch over the union of question–answer checks used by feasible proofs |

A feasible proof uses two distinct selected answers by the same
author, each in the depth-64 rank for its respective question.
This filter needs only the resident author index and ranks. It
does not examine answer text or model verdicts. The staged variant
must still verify eligible anchors for its third-source decision.
The sparse routes were added after the dense pilot. Their labels
and trace subset were kept fixed.

The static split arm isolates the cost of dividing one fixed plan
across GPU forwards. Its document identities and actual input token
counts must match the static batched arm, request by request.
Capture output differences to detect any decision sensitivity to the
reranker's batch shape.

Compare all five arms on the same open-loop random arrival trace at
each rate. Measure scheduled arrival through receipt of the whole
response, including queuing. Repeat on the same node with a different
trace seed and rotated run order. Record natural issuance and known
accepted-answer pair hits. Also match staged repair and static
batching to the smaller issuance count among shared completed
requests. The test set was constructed from questions with known
pairs, and the accepted-answer label is incomplete outside those
pairs.

## Results

In the dense pilot (Slurm array 46892), batching the same six
checks into one GPU forward preserved every proof. It increased
client P95 relative to static split in all four traces. The pilot
prompted the eligibility filter evaluated below.

Both jobs completed all requests. P95 below measures client latency
in milliseconds from scheduled arrival through the full HTTP response.

| Actual offered requests/s | Staged | Static split | Static batched | Staged sparse | Static sparse |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4.45 | 218 | 220 | 235 | **115** | 126 |
| 7.31 | 285 | 292 | 378 | 144 | **112** |
| 12.45 | 940 | 941 | 1,103 | 237 | **231** |
| 20.48 | 2,427 | 2,473 | 2,775 | 846 | **659** |

| Service arm | Checks per trace | Distinct excerpts used per trace | Verifier prompt tokens per trace | Known returned pairs |
| --- | ---: | ---: | ---: | ---: |
| Staged | 324 | 162 | 243,790 | 24 |
| Static split | 324 | 162 | 245,506 | 24 |
| Static batched | 324 | 162 | 245,506 | 24 |
| Staged sparse | 184 | 130 | 140,048 | 24 |
| Static sparse | 163 | 124 | 121,402 | 24 |

Batching alone reproduced the pilot result. It preserved each
static proof and increased P95 relative to static split. Filtering
preserved the full route's returned proof for all 54 requests, for
both static planning and staged repair. Static filtering used
49.7% fewer checks and 50.6% fewer prompt tokens than static batching.
The P95 reduction ranged from about 46% to 79%. These are
descriptive ranges including traces that queue substantially.

Relative to the filtered staged comparator, the filtered static
route used about 11% fewer checks, with lower P95 on three traces
and higher P95 at the lowest load. The two planning routes changed
which proof was returned on two requests. Each returned 24 known
proofs, including at matched issuance. The same 54 constructed
requests underlie both runs. This result establishes the serving
benefit of filtering on this scoped task. The distinct question
pairs do not provide a representative estimate of precision, and
mixed-source expert search still requires separate validation.

Slurm array 46899 ran on an RTX 4090 on ws1. The final dense control
proof and source signatures independently reproduced the dense pilot.
[Aggregate results](results.json).

## Reproduce

Restore the MathOverflow snapshot and the two pinned checkpoints
from the source-selection experiment. Use one Slurm GPU and a fresh
output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.community.expected_support_service \
  MATHOVERFLOW QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/community-joint-answer-witness-v1/results.json NEW_RESULTS
```

Run again with `--offset 1` and a different output path.
