# Measure equal-output source-access policies over loopback HTTP

Freeze each request's candidate pools and byline source plan using the
same live embedding and source snapshots as the
[three-read reference](../live-joint-source-request-v1).
Replay 422 arrivals, two fixed shuffles of the 211 PLOS and eLife
requests, in each policy and rate setting. Every source access
opens a loopback HTTP/1.1 connection and reads the original
local declaration bytes from the source service. The service adds
16 ms per read inside a semaphore with six concurrent slots.
Request timing starts at scheduled arrival and ends at response
serialization. It includes source queueing, HTTP exchange and proof
matching. The latency replay excludes query encoding and candidate
retrieval, which are held fixed outside the timed phase.

| Policy | Fetch schedule |
| --- | --- |
| All three in parallel | Issue all three reads at arrival |
| Byline precheck | Issue all potentially relevant reads in parallel. Skip the third when no ranked pairing connects it to either anchor through a shared author |
| Two-read certificate | Read the anchors in parallel. Fetch the planned third source only when it could change the winning expert and citations according to the observed roles and its byline |

The precheck operates before any reads. The certificate is evaluated
after both first sources arrive. It only needs the planned third
source for this fixed-output comparison. The
[pool-wide certificate](../joint-prefix-completion-v1) also rules out
alternatives outside the planned triple. Both require trusted bylines,
fixed ranks and a two-source proof for one person. The request cache
stores a reference response obtained by local verification of all
three sources. The load pass verifies every returned response
byte-for-byte and cross-checks its source counts against the service.

## Results on ws5

Both tasks of Slurm array `44597` completed. All 422 responses per arm
and rate match the fixed reference byte-for-byte in both runs. Read
counts and transferred declaration bytes reproduce exactly.

| Scheduled queries/s | Metric | All three in parallel | Byline precheck | Two-read certificate |
| ---: | --- | ---: | ---: | ---: |
| 60 | P95 latency (ms) | 18.63–18.68 | 18.19–18.23 | 32.53–32.55 |
| 165 | P95 latency (ms) | 961.42–976.25 | 24.67–24.69 | 33.09–33.30 |
| 165 | Drained queries/s | 118.17–118.66 | 164.07–164.11 | 164.28 |
| Both | Declaration reads | 1,266 | 910 | 866 |
| Both | Declaration payload bytes | 942,542 | 691,170 | 673,542 |

The precheck skips the third read on 127 of 160 PLOS requests and
all 51 eLife requests. It reduces source accesses by 28.1%. The
two-read certificate skips 44 more reads, covering 22 additional
PLOS requests that occur twice in the trace. On the 22 arrivals
that still need a third read, the certificate waits for a second
fetch wave. At 60 scheduled queries/s its P95 rises even though its
median stays close to the other policies. The derived third-source
certificate is useful when reducing read load matters more than the
latency of those requests.

At 165 scheduled queries/s the unpruned source service accumulates
a backlog. The precheck reduces the offered document traffic enough
for this bounded service to keep up. The resulting latency difference
is a capacity effect in a finite trace. The inserted delay and worker count set the operating point.
Production throughput and external storage latency require a separate
measurement. All fetches use loopback, fresh connections and
unchanged local declaration files.

## Reproduce

Restore the pinned publisher snapshots and model used by the
live request replay. Run twice on a Slurm compute node with a GPU
and fresh output paths. Use `--order-offset 1` for the second run.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.prefix_source_http \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json NEW_RESULTS
```

[Source access and timing results](results.json).
