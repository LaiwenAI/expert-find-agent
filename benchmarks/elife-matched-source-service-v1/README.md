# Test calibrated source fallback under sustained replay

Compare a single-slot baseline, a student-to-4B role cascade, the
cascade with dense source-only fallback, and the cascade with a
calibrated source gate followed by one 4B role check. The two
fallbacks run only if the cascade has neither selected a paper pair
nor spent its verifier forward.

Rebuild the [matched source gates](../elife-matched-source-calibration-v1)
from earlier eLife snapshots and freeze the calibration against the component
record. Route lowercase normalized topic clues of at most eight words to
the keyword threshold and other inputs to the excerpt threshold. The
HTTP request supplies the clues and the requested roles. Its label is
used only for evaluation and the format-agreement audit. No route uses a
target paper or label to make a serving decision.

Replay the [independent panel](../elife-joint-postings-service-v1)
and the [development panel](../paired-role-prospective-service-v2)
in both arm orders. Each order and rate uses one GPU worker and a
matched Poisson arrival trace. Repeat each distinct request eight times
per arm and rate. Client P95 includes queueing. Quality is scored once
per distinct request, with intervals over connected paper and recorded
author groups. Report role forwards and source forwards separately
because the two verifier models have different costs. All paths use at
most one verifier forward per request.

## Results

All tasks ran on RTX 5090 workers on ws3 in Slurm array 46338.
The format rule agrees with every constructed request view. Evidence
is deterministic across the arrival rates, order reversal, and
repeated requests, and matches the component reference for the
cascade and the two fallback arms.

### Held panel

Each replay sends 384 requests with 48 distinct inputs and 12 connected
paper and author groups. Each view contains 12 positives and 12 swapped-role
controls.

| Policy | Keyword strict / adverse | Excerpt strict / adverse | Role / source forwards per replay | Client P95 at 31.03 req/s (ms) | Client P95 at 66.50 req/s (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-slot baseline | 8 / 0 | 10 / 0 | 112 / 8 | 73.2–76.8 | 1174.1–1195.6 |
| Student to 4B | 10 / 0 | 12 / 0 | 128 / 8 | 86.8–92.0 | 1428.5–1443.3 |
| Dense source-only fallback | 11 / 1 | 12 / 0 | 128 / 24 | 88.3–93.8 | 1553.8–1625.5 |
| Calibrated source gate and 4B role check | 11 / 0 | 12 / 0 | 144 / 8 | 84.1–86.1 | 1650.9–1675.0 |

The calibrated path preserves the extra keyword proof and suppresses the
observed wrong-role issuance. The grouped descriptive interval for the
strict keyword change relative to the cascade is 0 to +25 percentage
points. The new path adds 16 role forwards per 384 requests and keeps
source forwards and rows at cascade levels. Against the dense fallback,
it trades 16 source forwards and 32 source rows for 16 role forwards.

At the higher rate, the new path raises P95 by 14.4%–17.3% against the
cascade and by 3.0%–6.2% against dense fallback in matched orders.
Completed throughput is 51.1 req/s for the new path and 53.0 req/s
for the cascade at the same 66.5 req/s offered load. Server-queue P95
exceeds 1.19 s for both. At the lower rate, the new path is 2.7–5.9 ms
faster than the cascade in two replays of one arrival trace.

### Development panel

Each replay sends 864 requests with 108 distinct inputs, from 27
linked paper pairs in 22 connected paper and author groups. Each view
contains 27 positives and 27 swapped-role controls.

| Policy | Keyword strict / adverse | Excerpt strict / adverse | Role / source forwards per replay | Client P95 at 29.79 req/s (ms) | Client P95 at 63.84 req/s (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-slot baseline | 10 / 0 | 12 / 0 | 96 / 32 | 54.0–54.6 | 505.1–607.9 |
| Student to 4B | 12 / 0 | 18 / 0 | 144 / 32 | 62.1–63.3 | 1105.2–1202.8 |
| Dense source-only fallback | 15 / 0 | 18 / 0 | 144 / 64 | 63.1–65.5 | 1446.4–1514.3 |
| Calibrated source gate and 4B role check | 14 / 0 | 18 / 0 | 168 / 32 | 62.9–67.7 | 1522.0–1570.8 |

The calibrated path gains two strict keyword proofs relative to the
cascade and gives up one of the three proofs added by the dense
fallback. It adds 24 role forwards per replay. At the higher rate,
its P95 increases by 30.6%–37.7% over the cascade and by 3.7%–5.2%
over the dense fallback in matched orders.

All 32 replays complete without request failures. The evidence
supports retaining the student cascade as the current low-cost path
and treating the calibrated role fallback as a quality candidate.
The new path exceeds cascade P95 under both higher-load traces.
A single-forward count conceals the extra 4B work. Future timing
comparisons should preserve evidence and test whether reducing or
batching that work improves throughput.

Both panels use previously analyzed structured requests. Eight repeats
increase the length of the load test without increasing the number of
independent quality observations. Zero observed adverse issuances in
the calibrated arm supports a larger risk test. The single-worker
loopback replay measures neither natural-language parsing nor
full-production latency. See the [aggregate results](results.json).

## Reproduce

Restore the projection, manifest, cutoff inputs, and Qwen checkpoints
from the [earlier service benchmark](../elife-joint-postings-service-v1#reproduce).
Restore both earlier eLife calibration snapshots and the matching
component record from the [source calibration benchmark](../elife-matched-source-calibration-v1#reproduce).
Use its original per-panel output or extract the corresponding object
from `results.json` under `panels.holdout` or `panels.development`. The
runner checks the calibration and proof signatures in that object, and
records the supplied file hash.
Run inside a Slurm GPU allocation for each evaluation panel. Repeat
with `--reverse` and a fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.credit.paired_role_prospective_service \
  PANEL_PROJECTION PANEL_MANIFEST benchmarks/paired-role-verification-v1/results.json \
  NISO_EXAMPLES ROLE_TRAINING SCORE_BANDS SOURCE_FROZEN ROLE_FROZEN \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B OUTPUT \
  --matched-source-service-reference PANEL_COMPONENT_RECORD --plos PLOS_PROJECTION \
  --matched-source-projection EARLIER_ELIFE_PROJECTION \
  --matched-source-manifest EARLIER_ELIFE_MANIFEST \
  --matched-source-supplement INITIAL_ELIFE_PROJECTION \
  --matched-source-supplement-manifest INITIAL_ELIFE_MANIFEST --repeats 8
```
