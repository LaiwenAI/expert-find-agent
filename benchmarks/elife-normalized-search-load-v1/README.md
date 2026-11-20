# Replay queued requests with shared role normalization

Reuse the 72 structured eLife queries, 280 articles with complete
contribution declarations, and model checkpoints from the
[serial agreement study](../elife-normalized-search-v1).
Compare one normalization per candidate source with one normalization
per request. Both use the same four-input agreement batch, read budget,
topic scorer and deterministic author-field check. This controls
the computation placement without changing the agent's evidence logic.
No results are cached across requests.

A local producer places requests on a FIFO queue using a seeded
sequence of exponential arrival gaps. Scale the same gaps to each
offered rate (2, 7 and 11 requests per second), and use the identical
query order and intended times in both arms. One worker with one
resident GPU performs all query work. Replay twice, changing the
arrival seed and reversing initial arm order in the second replay.
Measure time from scheduled arrival until the response is assembled
in the worker, actual queue time, service time and ingress lateness.
Count unfinished requests at the four arrival quartiles and the
final drain time. Check equality of the serialized ordered authors
and source references for each request.

The driver has no external protocol, network, or free-text request
parser. Its finite arrival profiles repeat one small request panel.
These experiments measure local queueing under specified schedules.
They do not estimate steady-state capacity or an external service SLA.

## Results

Both tasks in Slurm array 44186 completed on an RTX 5090.
All 432 pairs of response payloads are byte-identical. For
each rate and replay, both arms emit 96 role-supported slots
and 58 tag-backed slots, with zero wrong-role slots under the
declaration contract. A 72-request replay uses 239 versus 68
normalizer forwards, a reduction of 71.5%.

| Offered requests/s | Scheduled arrival to completion P95, per source | Scheduled arrival to completion P95, per request |
| ---: | ---: | ---: |
| 2 | 410–489 ms | 116–132 ms |
| 7 | 2.12–2.31 s | 186–189 ms |
| 11 | 4.93–4.99 s | 352–376 ms |

Ranges show the two arrival draws. At 7 requests/s, queue P95
drops from 1.93–2.15 s to 106–112 ms and service P95 from
309–311 ms to 84–85 ms. At 11 requests/s the baseline has
33 and 37 unfinished requests at the final arrival. Shared
normalization has four in both draws. Ingress timing jitter stays
near 0.1 ms at P95. The measurements isolate normalization reuse
against duplicated work; optimized topic and role-aware systems
should share this execution improvement when comparing retrieval
quality.

[Both GPU runs](results.json)

## Reproduce

Use the verified eLife projection, manifest and checkpoints from
the serial benchmark in an allocated Slurm GPU.

```sh
export PYTHONPATH=src:.
python -m experiments.credit.elife_normalized_search_load PROJECTION \
  benchmarks/elife-role-nomination-v1/manifest.json \
  benchmarks/elife-role-nomination-v1/results.json NEW_RESULTS --replay 0
```

Repeat with `--replay 1` and a fresh output path.
