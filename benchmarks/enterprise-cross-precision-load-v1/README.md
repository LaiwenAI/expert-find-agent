# Load replay for selective cross-precision expert search

The [serial search benchmark](../enterprise-cross-precision-search-v1/README.md) found a lower warm first-query P95 and identical reference outputs. This experiment measures whether queueing magnifies or absorbs that difference. The two arms are the existing BF16 2.0-logit guard and selective cross-precision confirmation with independent BF16 fallback. The models, search index and evidence verifier are shared. Each arm gets a warm-up before the timed workload.

Replay all 80 development requests once per trace in a seeded random order. Independent exponential interarrival intervals model open-loop load. Compare nominal rates of 1.5 and 3 requests/s, with two shared seeds at each rate. Reverse arm order on the second seed. Every client issues its request according to the planned schedule even when earlier requests remain queued. The server binds to loopback and uses one FIFO GPU search worker. This compares many outstanding callers using one shared worker. It does not compare GPU request batching or run across hosts.

For each arm and trace, measure client P50/P95/P99 from planned release to receipt, server queue and service times, client release lag, completed throughput and queue drain after the final scheduled release. Compare the serialized qualification spans, candidate order, ranked experts, citations and unresolved states per request. Treat the four trace pairs as descriptive repetitions. Requests in one queue have dependent latencies.

## Results

| Nominal rate | Seed | Client P95, BF16 → selective | Server queue P95, BF16 → selective | Drain, BF16 → selective |
| ---: | ---: | ---: | ---: | ---: |
| 1.5/s | 20261119 | 8.64 → 4.33 s | 7.19 → 2.88 s | 0.27 → 0.19 s |
| 1.5/s | 20261120 | 6.50 → 3.38 s | 6.08 → 2.95 s | 5.78 → 0.36 s |
| 3.0/s | 20261119 | 16.02 → 6.85 s | 15.76 → 6.47 s | 15.35 → 5.45 s |
| 3.0/s | 20261120 | 9.74 → 5.04 s | 9.26 → 4.17 s | 12.64 → 5.60 s |

Client P95 falls 48.1–57.2% across the four paired traces. All 320 paired responses have the same serialized qualification spans, candidate order, ranked experts, citations and unresolved states. At nominal 3/s, batch completion throughput rises from 1.96–2.01 to 2.44–2.58 requests/s, measured from the first planned arrival to the last completion. Both arms drain a queue after the final release at this rate. These finite bursts estimate latency under the tested workload. They do not establish sustainable serving capacity, independent expert relevance, or cross-host performance.

Job 41493 ran on a ws5 RTX 5090. No response text or request-level rows are included in the release. [Aggregate results](results.json)

## Reproduce

Use the pinned development query and source projections and checkpoints from the linked serial benchmark. Run on a CUDA Slurm compute node with a new result path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_cross_precision_load \
  DEVELOPMENT_QUERIES DEVELOPMENT_SOURCES CHECKPOINT_DIRECTORY NEW_RESULTS
```
