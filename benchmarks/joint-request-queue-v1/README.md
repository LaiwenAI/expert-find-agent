# Measure queueing and output parity under live joint requests

Replay the 160 PLOS and 51 eLife heading requests from the
[live structured request benchmark](../live-joint-source-request-v1).
The workload has 211 requests with both publisher indexes ready.
Bodies contain the two controlled topic–role requirements.
Each request gets live embeddings, hybrid source ranking, joint
nomination, three actual declaration file reads and serialized
source-backed results. Offline article encodings and index build
stay outside the measurement. The route uses original author
identifiers and structured roles rather than a generative verifier.

Freeze the query token width at 64, with left padding and position
IDs relative to the attention mask. Model weights, source ranking
and the response rule are shared. For the reference, independently
encode the two slots of each request and save its exact response
bytes. That reference is used only for evaluation. Replay repeated
live request computation in all measured arms.

| Arm | Worker dispatch |
| --- | --- |
| Serial | One request per GPU forward with no coalescing timer |
| Batch 4 | Batch up to four already-arrived requests. Wait up to 2 ms from the oldest request's arrival if the batch is not full |
| Batch 8 | Same deadline, with up to eight requests in a batch |
| Padded 8 | Same dispatch, with a fixed tensor of 16 topic rows. Fill unused rows with copies of the first request and discard their outputs |

Use one worker and Poisson arrival gaps from a fixed seed, with
nominal rates of 60, 120 and 240 requests/s. Every route gets the
same shuffled requests and scheduled arrival times. Each arrival
enters the queue by its scheduled time even when the worker is
busy. The worker waits against the real wall clock and processes
the complete trace. Report exact response mismatches against the
single-request layout and a second independent reference made by
running each request alone in the 16-row layout. The latter isolates
response changes due to peers from changes due to shape. Report
batch-size counts, achieved completion
rate across the run and time from scheduled arrival to serialized
response. Queueing includes the coalescing timer. The trace runs
until the worker drains, so a high nominal offered load is not
equivalent to a sustained completion rate.

This is a local synthetic load replay. It includes actual model
inference, candidate assembly, warm local file reads and in-process
queueing. It excludes network, transport, real client or HTTP
thread overhead, external storage and other service traffic.
Short canonical queries fit the chosen width. It does not
measure free-form parsing or require a topic model.

## Results

The two tasks of Slurm array `44508` ran on RTX 5090 on ws5 and
rotated execution order. The short trace's realized arrival rates
were 71.4, 142.8 and 285.6 requests/s. The table reports time from
scheduled arrival to serialized response, including queueing.

| Realized arrival requests/s | Serial P95 (ms) | Batch 4 P95 (ms) | Batch 8 P95 (ms) | Padded 8 P95 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 71.4 | 37.45–37.56 | 18.21–18.42 | 18.36–18.50 | 23.92–24.16 |
| 142.8 | 543.47–545.39 | 19.23–19.42 | 19.49–19.61 | 25.10–25.48 |
| 285.6 | 1231.81–1262.93 | 24.20–25.44 | 20.54–20.93 | 25.94–26.12 |

At the top load, serial drains at 102.1–103.8 requests/s and
batch 8 drains at 282.0–282.1. Serial queue P95 exceeds 1.2 s.
This measures draining a short overloaded trace, not the
production tail at comparable utilization. Batch 8 returned
exactly the same response bytes as the 2-row layout in both
runs. The initial unpadded pilot in array `44503` changed one
batch-4 response at the top load in one of two runs, while
batch 8 had no observed changes. Wall-clock execution changes
the batch groups. No finite replay establishes a general
bitwise guarantee for adaptive batch sizes.

Padding to 16 rows reproduces the independent 16-row response
for every request. One reference response differs between the
2-row and 16-row layouts, and that response remains different
in the padded replay. Fixed-shape comparison must name its
reference layout. The padded route adds about 5 ms P95 against
dynamic batch 8 at this workload's top load.

[Aggregates and batch-size counts](results.json)

## Reproduce

Restore the pinned PLOS and eLife snapshots and Qwen3 embedding
checkpoint from the live structured benchmark. Run within a
Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.joint_request_queue \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING NEW_RESULTS
```

Repeat with `--order-offset 1`.
