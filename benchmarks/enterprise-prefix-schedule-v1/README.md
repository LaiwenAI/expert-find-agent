# Test shared-prefix expert search under queued HTTP requests

The [serial enterprise test](../enterprise-prefix-fork-v1) measured a modest
first-query tail reduction after sharing prefix storage. This experiment
tests whether the gain survives open-loop demand and measures a separate
request-ordering policy on the same inference path.

Use the same 80 development requests, 52,667 owned sources and pinned models.
Build the exact index once. A loopback HTTP service uses one GPU worker and
recomputes each request. There is no cross-request cache, preemption, or GPU
batching. Compare three arms:

| Arm | Prefix cache | Ready request |
| --- | --- | --- |
| Copy FIFO | Copy and repeat prefix tensors within each clause batch | Oldest arrival |
| Broadcast FIFO | Broadcast the same read-only prefix tensors | Oldest arrival |
| Broadcast with aging | Broadcast the same read-only prefix tensors | Shortest request by character count until the oldest ready request has waited 750 ms, then serve the oldest |

The aging threshold affects the choice at a dispatch opportunity and does
not put a bound on waiting behind already running work. Character count is
a cheap input-side proxy. Work on general LLM serving uses learned relative
runtime estimates for priority scheduling.
[Fu et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html)

Run two 80-request traces with distinct frozen seeds and nominal exponential
arrivals at 3 requests/s. Reuse arrival times and request permutations
within each trace. Reverse arm order on the second trace. Compare the
serialized qualification spans, retrieved experts, ranking, citations and
unresolved states. Client delay is measured from scheduled release, including
the queue and loopback transport. Record the longest 20 requests by input
character count separately. Queueing makes timings within a trace dependent.

## Results

Slurm job 42944 ran on an RTX 5090. All 480 HTTP requests completed. For each
seed, all 80 responses from each new arm are byte-identical to the copy FIFO
responses. Actual offered rates are 3.09 and 2.90 requests/s.

| Seed | Arm | Client P50 / P95 | Service P95 | Longest quartile P50 / P95 | Drain |
| --- | --- | ---: | ---: | ---: | ---: |
| 20261119 | Copy FIFO | 1.16 / 6.78 s | 1.180 s | 1.22 / 6.18 s | 5.37 s |
| 20261119 | Broadcast FIFO | 1.05 / 6.56 s | 1.137 s | 1.13 / 5.86 s | 5.03 s |
| 20261119 | Broadcast with aging | 1.01 / 6.55 s | 1.137 s | 1.33 / 5.85 s | 5.02 s |
| 20261120 | Copy FIFO | 2.07 / 5.04 s | 1.179 s | 1.96 / 5.50 s | 5.59 s |
| 20261120 | Broadcast FIFO | 2.00 / 4.94 s | 1.134 s | 1.90 / 5.33 s | 5.39 s |
| 20261120 | Broadcast with aging | 1.78 / 4.94 s | 1.132 s | 1.83 / 5.35 s | 5.41 s |

With FIFO fixed, broadcast lowers service P95 by 42–45 ms and client P95
by 103–226 ms, or 2.0–3.3%. It leaves a queue to drain on both traces.
Under shared-prefix execution, aging changes client P95 by −7 and +8 ms.
It lowers overall median delay in both traces, while in the first trace
it increases the median for the longest quartile by 195 ms. This shows
the redistribution of wait time across request sizes. The timing evidence
supports broadcast as an execution improvement on this workload. It does
not support priority scheduling as a general tail-latency improvement.

The two traces reuse the development cohort. They characterize finite bursts
on one GPU worker and do not establish sustainable capacity, behavior under
a different input mix, or independent expert-matching quality.

[Aggregate results](results.json)

## Reproduce

Use the pinned development corpus and checkpoints from the
[serial benchmark](../enterprise-prefix-fork-v1). Run in a Slurm CUDA
allocation with the serial aggregate and a new output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_prefix_schedule \
  DEVELOPMENT_QUERIES DEVELOPMENT_SOURCES CHECKPOINT_DIRECTORY SERIAL_RESULTS NEW_RESULTS.json
```
