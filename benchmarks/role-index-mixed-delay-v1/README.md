# Parallel source reads with heterogeneous access delay

Use the same held requests, quarter-capacity index, two HTTP
services and exact reference responses as the
[uniform-delay replay](../role-index-read-waves-http-v1).
Four persistent clients run the workload with three delay maps.
Each map assigns either zero or two added milliseconds per
source. Choose delayed sources by seeded sampling of the sorted
source IDs, independently of the queries and proof results.

- **Sparse known** assigns added delay to 25% of sources.
  Only the adaptive policy sees the assigned map.
- **Dense known** assigns added delay to 75% of sources.
  The adaptive policy likewise sees the assigned map.
- **Shifted known and shifted stale** reassign the delayed
  25% with a new seed. Hold the actual delay fixed in the two
  replays. The adaptive policy sees the new map in the known
  arm and the original sparse map in the stale arm.

At each frontier, the adaptive policy considers the same first
four unresolved sources as the fixed width-four policy. It
fetches them together when the sum of their expected added
delays minus the largest delay is at least 0.75 ms. Otherwise it
reads only the leading source. With these binary delay maps
the rule opens a parallel wave if at least two selected sources
are predicted slow. The rule and threshold are fixed before
the replay. Known-map results give the policy perfect metadata
about the injected delay. Real transport and fitting an
estimator from production telemetry are separate tasks.

Each response must match the original exact ranking and
minimum citations byte-for-byte. Verify the fetch set for each
request, total bytes, and the fixed policies' read-round counts.
Use two opposing policy orders per publisher.

## Results

The four completed tasks of Slurm array `45176` reproduce
312 exact responses per publisher and policy in every profile.
All policies fetch exactly the same sources for each request.
Client P95 ranges in milliseconds span two policy orders.

| Publisher and delay map | Serial | Fixed width four | Adaptive |
| --- | ---: | ---: | ---: |
| PLOS, sparse known | 10.18–10.74 | 10.29–10.40 | 9.93–10.41 |
| PLOS, dense known | 20.95–20.98 | 16.27–16.30 | 16.32–16.35 |
| PLOS, shifted known | 7.98–8.19 | 8.44–9.11 | 8.53–8.71 |
| PLOS, shifted stale | 8.13–8.46 | 8.72–8.98 | 8.72–8.93 |
| eLife, sparse known | 18.67–19.28 | 19.30–19.37 | 17.42–17.98 |
| eLife, dense known | 35.56–36.16 | 26.11–26.51 | 27.50–27.97 |
| eLife, shifted known | 15.97–16.45 | 19.05–19.50 | 17.51–17.91 |
| eLife, shifted stale | 15.77–16.11 | 18.21–18.81 | 16.38–16.97 |

The adaptive policy reduces P95 by about 7% relative to serial
on the first sparse eLife map. Fixed width four is faster with
dense delay, and serial is faster under both shifted sparse
replays. At fixed actual delay, stale metadata raises adaptive
P95 in PLOS and lowers it in eLife. This rule does not
dominate the fixed controls even with knowledge of the
artificial delays. Dependence on which needed sources are
slow is stronger evidence than the catalog-wide slow fraction.
[Per-workload aggregates and all controls](results.json).

## Reproduce

Use the command for the
[uniform-delay HTTP benchmark](../role-index-read-waves-http-v1),
adding `--mixed-delay`. Repeat with `--order-offset 1`.
