# Transport test for parallel exact source reads

Replay the same 312 held role requests per publisher from the
[read wave benchmark](../role-index-read-waves-v1). Fix the quarter
capacity index to whole cold placement for PLOS and smoothed
role columns for eLife, as supported by the
[sequential HTTP comparison](../role-column-index-http-v1).
All four policies share the same index within a publisher.
Compare the existing owner-by-owner serial policy with the
shared frontier at widths one, two and four.

Both services run on one Slurm CPU node. The source endpoint
opens and returns the materialized JSON for each fetch. For
parallel waves, a search connection keeps up to four persistent
source connections and overlaps requests with worker threads.
Warming uses four requests per search connection. Each timed
request includes client-to-search HTTP, proof planning, source
reads, and serialization. The source endpoint adds zero or two
milliseconds per fetch under the stated workload. One and four
clients send requests in closed loop. The study uses injected
delay and does not measure open-loop service capacity.

Require byte-for-byte equality with the exact reference JSON.
Check source payload bytes and counts against the serial run,
verify the read round counts against the CPU benchmark, and
compare the ordered sequence of per-request source-set hashes
across policies. Repeat with reversed policy order.

## Results

All four tasks in Slurm array `45149` reproduce the exact
response bytes, per-request source sets, total bytes and read
round counts. The table shows client-observed P95 in milliseconds
across two policy orders.

| Publisher and request profile | Owner serial | Frontier width 1 | Frontier width 2 | Frontier width 4 |
| --- | ---: | ---: | ---: | ---: |
| PLOS, 1 client, 0 ms added | 1.10–1.11 | 1.23–1.25 | 1.46–1.48 | 1.52–1.54 |
| PLOS, 4 clients, 0 ms added | 6.58–6.64 | 7.21–7.43 | 7.36–7.44 | 7.39–7.59 |
| PLOS, 4 clients, 2 ms added | 23.72–23.87 | 23.88–23.96 | 18.14–18.41 | 18.05–18.36 |
| eLife, 1 client, 0 ms added | 2.42–2.45 | 2.79–2.82 | 3.19–3.21 | 3.28–3.33 |
| eLife, 4 clients, 0 ms added | 13.37–13.83 | 14.78–15.18 | 16.35–16.67 | 15.50–16.82 |
| eLife, 4 clients, 2 ms added | 54.91–55.16 | 55.72–56.06 | 35.83–36.27 | 32.99–34.05 |

At four clients and 2 ms added per source, width four reduces
P95 by 22.6–24.4% on PLOS and 38.3–39.9% on eLife. Width two
already realizes most of the PLOS gain. With no added delay,
the serial control is faster for every configuration. These
measurements motivate a deployment policy that accounts for
source delay and fan-out contention. The study does not
identify a universal threshold or measure full free-text
expert search. [Aggregate outputs](results.json).

## Reproduce

Use a Slurm CPU allocation, the pinned publisher projections,
and the prior results.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.role_column_index_http \
  CORPUS benchmarks/elife-role-nomination-v1/manifest.json \
  benchmarks/role-column-index-v1/results.json SCRATCH NEW_RESULTS \
  --publisher elife --waves benchmarks/role-index-read-waves-v1/results.json
```

For PLOS, supply its source fixture and pass `--publisher plos`.
Repeat with `--order-offset 1`.
