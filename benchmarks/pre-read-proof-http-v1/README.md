# Measure bounded HTTP source access on the frozen eLife plans

Use the same 398 requests, packed role index and depth-128 three-source
plans as the [local pre-read study](../pre-read-proof-bound-v1).
Prepare the two visibility profiles with the frozen GPU encoder before
starting timed dispatch. Each trial issues 398 arrivals at a fixed rate.
All four methods see the same arrival order within a repeat. Reverse
the arrival, policy, visibility and rate orders in the second repeat.
For every arrival, compare the actual response bytes against local
verification over all three planned declarations.

The loopback HTTP/1.1 source service opens the original projected
declaration for each fetch and adds 16 ms inside a six-slot semaphore.
Record actual reads, transferred bytes, fetch waves and client P95 from
scheduled arrival through serialization. Track arrival-release lag and
the last request's drain separately. Expand the process file descriptor
limit to hold connections during intentional overload. Query encoding,
candidate retrieval and plan construction happen outside the timer.
There is no external traffic, and rates describe a finite replay.

| Indexed roles | Arrivals/s | Parallel three | Fixed two | Byline precheck | Pre-read role bound |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 60 | 20.77–22.23 | 20.41–21.63 | 20.40–21.67 | 20.20–20.75 |
| 100% | 165 | 904.13–983.19 | 19.31–19.82 | 21.59–23.11 | 19.41–20.95 |
| 100% | 205 | 1,383.12–1,416.47 | 297.49–301.88 | 301.42–327.08 | 23.64–24.70 |
| 50% | 60 | 21.77–22.17 | 20.57–21.74 | 20.75–21.80 | 21.40–22.37 |
| 50% | 165 | 924.04–935.74 | 18.15–19.38 | 21.74–22.00 | 46.58–59.52 |
| 50% | 205 | 1,436.31–1,517.25 | 301.35–334.39 | 324.99–360.82 | 328.83–328.91 |

P95 values are in milliseconds. With full role visibility the
pre-read bound serves 526 reads against 796 for fixed two. At 205/s
fixed two sends approximately 410 source reads/s to a six-slot service
whose nominal delay-only capacity is 375 reads/s. With partial
visibility the pre-read bound sends 791 reads in fewer waves
and still experiences queueing. At 165/s, serving some requests
with three parallel reads produces a larger tail than fixed two,
despite similar aggregate read volume. The strong two-read control
happens to recover every response on both panels. The conclusions
concern source service contention at the measured load and delay.

Slurm job 47568 completed on ws6 with an RTX 4090. Both orders return
all 398 responses per arm and setting byte-for-byte. Reads, waves and
transferred source bytes reproduce. The code for this run is pinned at
`2e51b98`. See the [complete receipts](results.json).

## Reproduce

Restore the frozen projections and checkpoint from the
[pre-read study](../pre-read-proof-bound-v1#reproduce). Run inside a
Slurm GPU allocation and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.pre_read_proof_http \
  EXTENDED_PROJECTION benchmarks/elife-extended-role-holdout-v1/manifest.json \
  CONFIRMATION_PROJECTION benchmarks/elife-source-guard-confirmation-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/pre-read-proof-bound-v1/results.json NEW_RESULTS
```
