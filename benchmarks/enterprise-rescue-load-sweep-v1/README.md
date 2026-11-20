# Measure queue buildup for the same enterprise rescue policy

Compare canonical padded verification with guarded variable-length
verification in the [existing loopback service](../enterprise-packed-role-load-v1).
Both paths run the same first-query retrieval, parsing and global role
rescue on abstention. The guarded reader checks its confidence band and
can rerun the original padded batch. Use one resident GPU and one FIFO
worker. Count all tokens and reruns. Verify response-byte equality
between the two paths, including source citations and parsed requirements.

At each rate (2.2, 2.7 and 3.2 requests/s), replay all 80 development
requests twice. Generate a different seeded permutation for each pass.
Use the same exponential gap sequence across rates and rescale it to
the exact mean offered rate. Pair the full order and schedule across
arms. Alternate execution order across rates. A single warm-up
request per arm is excluded. No sealed requests enter this benchmark.

Measure client, queue and service p95. At each arrival quartile
report the number of scheduled requests still unfinished, including
the new arrival, and the queue p95 of that quartile. Distinguish
completion rate during arrivals from completion rate after drain.
A finite schedule can end with a short queue because arrivals are
bursty. Growth over several quartiles is more informative than an
isolated nonzero drain. The experiment uses a single arrival draw,
repeats one development population and gives finite traces rather
than a steady-state capacity estimate.

## Results

Slurm job `43589` completed 960 responses on one RTX 5090.
All 480 paired responses were byte-identical. Each arm issued
74 provisional responses per rate. Guarded execution uses
842,316 verifier tokens per trace, including rechecks, against
1,171,380 in the padded route (28.1% fewer).

| Offered requests/s | Arm | Client P95 | Queue P95 | Service P95 | Unfinished at last arrival | Final drain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2.2 | Padded | 6.093 s | 5.421 s | 1.139 s | 4 | 0.924 s |
| 2.2 | Guarded | 4.990 s | 4.277 s | 1.063 s | 4 | 0.764 s |
| 2.7 | Padded | 8.496 s | 8.066 s | 1.144 s | 7 | 1.634 s |
| 2.7 | Guarded | 7.313 s | 6.780 s | 1.067 s | 4 | 0.894 s |
| 3.2 | Padded | 10.913 s | 10.587 s | 1.148 s | 25 | 8.056 s |
| 3.2 | Guarded | 9.022 s | 8.762 s | 1.072 s | 18 | 4.527 s |

The guarded reader reduces client P95 by 18.1%, 13.9% and
17.3% on these traces. Bursty arrivals still cause multi-second
queues at 2.2 requests/s. At 3.2 requests/s, unfinished counts
at the four arrival quartiles are 1, 16, 30, 25 for padded and
1, 16, 29, 18 for guarded. Observed completion rates during
the arrival window are 2.72 and 2.86 requests/s, respectively.
Both paths have substantial unfinished work at the end of
arrivals. Those observations do not establish a stable service
capacity at 3.2 requests/s. The timing comparison holds outputs
fixed in the reused development sample. It does not measure
expert relevance or cost per independently validated success.

[Aggregate timings and model work](results.json)

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_rescue_load_sweep \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS ROLE_WINDOWS SOURCE_VECTORS VECTOR_METADATA \
  MODELS benchmarks/enterprise-packed-role-search-v1/results.json NEW_RESULTS
```
