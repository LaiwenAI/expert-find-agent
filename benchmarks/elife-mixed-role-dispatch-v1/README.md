# Dispatch single and paired role checks together

The [paired-only comparison](../elife-matched-microbatch-v1) grouped
few 4B calls and held completed requests until all of their queue
peers had finished. Keep embedding and search planning per request.
Drain up to four already-queued HTTP requests in FIFO order. As each
plan finishes, release requests that need no 4B verifier. Collect both
one-prompt and two-prompt role checks for the remaining requests and
evaluate them together. Each request retains its own prompt and
options, margin threshold, and source evidence.

A single pending role check uses its original execution path. A mixed
forward can carry up to eight prompts. Slice the scores back by
request and apply the unchanged acceptance rules. Count logical role
checks, physical 4B forwards, and the composition of each shared
forward. Keep source checking per request.

Cross serial and mixed dispatch with the lightweight cascade and the
calibrated source fallback. Replay the [same arrivals](../elife-matched-source-service-v1)
on the previously analyzed 48-request held and 108-request development
panels. Present each request eight times per rate, reverse arm order
in the second run, and measure scheduled arrival to client response.
A replay is eligible only when repeated requests, offered rates, and
both schedules reproduce the archived evidence signatures.

## Results

### Held panel

Both orders reproduce the archived evidence on all 48 distinct
requests, including 10/12 strict keyword proofs for the cascade and
11/12 for calibrated fallback. Both produce 12/12 strict excerpt
proofs with zero observed adverse issuances.

| Arrival rate (req/s) | Search policy | Serial P95 (ms) | Mixed P95 (ms) | Physical 4B forwards per 384 requests |
| ---: | --- | ---: | ---: | ---: |
| 31.03 | Cascade | 85.4–86.7 | 82.1–83.8 | 128 → 119–122 |
| 31.03 | Calibrated fallback | 84.5–95.1 | 85.0–96.0 | 144 → 131–134 |
| 66.50 | Cascade | 1470.3–1515.9 | 1200.4–1215.9 | 128 → 78–82 |
| 66.50 | Calibrated fallback | 1645.3–1703.6 | 1404.0–1411.0 | 144 → 85 |

At the higher rate, mixed dispatch lowers P95 by 17.3%–20.8% for
the cascade and 14.2%–17.6% for the fallback in matched orders. The
calibrated fallback shares work in 45–46 4B forwards, including both
single- and paired-prompt checks. Logical verification work and
nonpadding verifier tokens are unchanged. With both policies using
mixed dispatch, the fallback keeps its extra strict keyword proof
and has 16.0%–17.0% higher P95.

### Development panel

Both orders reproduce the archived evidence on all 108 distinct
requests. The cascade and calibrated fallback retain 12/27 and
14/27 strict keyword proofs, respectively. All 32 service replays
across both panels finish with no request failures.

| Arrival rate (req/s) | Search policy | Serial P95 (ms) | Mixed P95 (ms) | Physical 4B forwards per 864 requests |
| ---: | --- | ---: | ---: | ---: |
| 29.79 | Cascade | 60.9–61.7 | 63.1–63.3 | 144 → 141–142 |
| 29.79 | Calibrated fallback | 65.9–67.1 | 66.5–67.8 | 168 → 161–162 |
| 63.84 | Cascade | 1204.1–1297.0 | 1035.2–1074.3 | 144 → 112–113 |
| 63.84 | Calibrated fallback | 1592.5–1705.7 | 1349.3–1358.0 | 168 → 126–128 |

At the higher development rate, the cascade and fallback P95 fall
by 14.0%–17.2% and 14.7%–20.9%, respectively. The observed P95
changes at lower rates are within 4% in both panels. With both
policies using mixed dispatch, calibrated fallback has 25.6%–31.2%
higher development P95 while retaining two additional strict keyword
proofs. The high-rate benefit comes from shared service execution
and applies to both policies. The experiment combines wider
verification eligibility and early completion, so it measures their
joint effect. Observed rates characterize a finite single-worker
loopback trace with a drain phase. Repeated requests lengthen the
load test without adding independent quality observations.

[All four runs](results.json) record role batch composition,
verification counts, throughput, queueing and proof signatures.

## Reproduce

Use the [calibrated fallback benchmark](../elife-matched-source-service-v1#reproduce)
with the same frozen inputs and GPU runtime. Add
`--mixed-role-batch-capacity 4` and run both arm orders for each panel.
