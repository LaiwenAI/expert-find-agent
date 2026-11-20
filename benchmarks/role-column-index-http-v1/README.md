# Serve exact structured expert proofs through two HTTP endpoints

Use the same PLOS and eLife projections, byte-capped indices,
and held-role requests from the [role-column benchmark](../role-column-index-v1).
Compare the whole-article author prior, whole-article cold-read
allocation, and smoothed individual role columns at the
original 25% packed-byte cap. Placement and model-free
query ranking are frozen from the previous benchmark.

The client sends three role strings over a persistent HTTP
connection. The search endpoint validates the roles, evaluates
the exact top-five proof, and serializes the response.
When the resident index cannot certify a result, search
loads the needed publisher declaration through a different
HTTP endpoint. The source endpoint reads and returns the
actual materialized JSON. Both endpoints are on the same
Slurm node, with one source connection per search client
connection. Keep all client connections warm before timing.

Each publisher contributes 312 requests across four held-role
folds. Measure one client with zero added delay, then four
closed-loop clients with zero or 2 ms added delay at each
source response. The 2 ms is an experimental intervention.
The service does not emulate internet or production I/O.
P95 is descriptive across requests. Overlapping folds are
not independent. There is no model inference, open-text parsing,
authentication, external source fetch or learned retrieval
ranking in the service path.

## Results

All four CPU tasks in Slurm array `45058` on `ws3` complete
and reproduce the exact expert order, citation bundles and
source byte counts from the original local replay. Placing
each request's expected response outside the timer lets
the client check the serialized JSON byte-for-byte.
Arm order rotates between the two tasks for each publisher.

| Publisher and profile | Whole author P95 | Whole cold P95 | Smoothed columns P95 |
| --- | ---: | ---: | ---: |
| PLOS, one client | 1.34–1.49 ms | 0.94–1.10 ms | 1.03–1.19 ms |
| PLOS, four clients | 8.18–8.57 ms | 5.97–6.05 ms | 6.45–6.57 ms |
| PLOS, four clients and delay | 34.52–34.77 ms | 23.76–23.87 ms | 26.31–26.40 ms |
| eLife, one client | 2.98–3.38 ms | 2.54–2.83 ms | 2.10–2.41 ms |
| eLife, four clients | 17.84–18.07 ms | 14.22–14.29 ms | 12.34–12.35 ms |
| eLife, four clients and delay | 80.31–80.53 ms | 68.56–68.57 ms | 55.02–55.36 ms |

PLOS has 1,744, 1,173 and 1,500 source reads, respectively.
eLife has 2,991, 2,663 and 2,438. On eLife the column allocation
reduces the four-client P95 by about 13% without the delay and
19–20% with it, relative to whole cold placement at equal bytes.
On PLOS, whole cold placement is faster than columns under
both concurrent profiles. Single-client PLOS timings overlap
and do not give a stable comparison between the two.
This result reproduces a publisher-dependent index trade-off
through actual loopback HTTP. It does not establish a
free-text enterprise service gain.
[Raw aggregate timings and fetch counts](results.json).

## Reproduce

Use the previous study's hashed publisher projections and
result manifest on a Slurm CPU allocation. Repeat with
`--order-offset 1` and a fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.role_column_index_http \
  CORPUS benchmarks/elife-role-nomination-v1/manifest.json \
  benchmarks/role-column-index-v1/results.json SCRATCH NEW_RESULTS --publisher elife
```
