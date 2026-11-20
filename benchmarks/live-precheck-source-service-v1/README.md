# Measure source prechecks through live retrieval

Use the earlier [HTTP source-access reference](../prefix-source-http-v1),
with six concurrent source slots and an inserted 16 ms per declaration
read. The source service sends the pinned publisher declaration bytes
over loopback HTTP. Compare an all-three parallel read with an eager
byline precheck that skips the third source only when it cannot form
a ranked pair with either anchor.

This experiment removes the cached-ranking assumption. Each arrival
queues a JSON request and undergoes fresh Qwen3-Embedding-0.6B inference,
hybrid ranking, contributor postings nomination, actual source HTTP
reads, declaration matching, and response serialization. Both policies
use one GPU worker, up to eight requests per batch, a 2 ms hold, and
the same padded shape of sixteen 64-token topic rows.
A dedicated executor runs planning while source exchanges progress.
The CPU proof and I/O code is shared between the policies.
No rank, plan or response is reused in the measured path.

Each arm runs two fixed shuffles of the same 211 PLOS and eLife
requests. A separate padded, single-request pass supplies the
reference response bytes from the original three declarations.
The trace verifies every issued or abstained response against that
reference. Timing starts at scheduled arrival and includes the
planner queue, both keyword and dense retrieval, HTTP source queue
and response serialization. The GPU and retrieval timing is the
full per-request wait from batch dispatch to ready plans. Stage
percentiles are computed separately and do not add to total P95.
There is no front-end network hop. The source delay and concurrency
are controlled, and the trace does not measure a production service.

## Results on RTX 5090

Both tasks of Slurm array `44604` on ws5 completed. Each policy
returns 422 byte-identical responses per rate and repeat, with the
same declaration count and payload bytes across repeats.

| Scheduled queries/s | Metric | All three in parallel | Byline precheck |
| ---: | --- | ---: | ---: |
| 60 | Total P95 (ms) | 33.17–33.29 | 33.12–33.21 |
| 165 | Total P95 (ms) | 977.32–981.09 | 52.20–53.02 |
| 165 | Source and matching P95 (ms) | 957.09–961.44 | 34.15–34.66 |
| 165 | GPU and retrieval P95 (ms) | 14.29–14.38 | 13.93–14.10 |
| 165 | Drained queries/s | 117.80–117.89 | 162.38–162.46 |
| Both | Source reads | 1,266 | 910 |
| Both | Declaration payload bytes | 940,382 | 687,906 |

The precheck saves 28.1% of reads. At 60 scheduled requests/s, both
policies complete at about 60 requests/s and have similar latency.
At 165 requests/s, the all-three source service builds a backlog.
The precheck reduces source traffic enough to avoid that backlog in
the same short trace, while planner queue P95 remains about 14 ms.
The high-load difference therefore reflects effective source capacity
under the inserted delay. Measurements from a sustained deployment
with real source access remain necessary.

The precheck count agrees with the earlier frozen-rank transport
experiment. The declaration byte totals differ because the earlier
reference used a different query encoding layout. The relevant
response parity here is against the same padded layout for each arm.

## Reproduce

Restore the pinned publisher snapshots and embedding model used in
the live request benchmark. Run on a Slurm GPU with fresh output
paths. Repeat with `--order-offset 1`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.live_precheck_source_service \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json NEW_RESULTS
```

[Complete timing and reads](results.json).
