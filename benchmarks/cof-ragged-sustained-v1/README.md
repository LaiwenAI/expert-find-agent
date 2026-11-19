# Longer load replay with shared-weight packed evidence verification

The [concurrent benchmark](../cof-ragged-concurrent-v1) established a finite-trace latency advantage using separate backbone copies and audited parameter sharing in a small follow-up. Here both evaluations use the shared-weight implementation. Packed and canonical model configurations remain separate. Parameter pointers are checked before the run. The GPU broker serializes live query embedding, full-inventory profile-and-source retrieval, and source scorer calls. Four request workers run the same two-citation frontier and use the same canonical fixed-shape recheck.

Compare padded four-row waves with independent, variable-length eight-row waves. Keep the original single-row path when a packed wave has one row. Cross-request merging is disabled.

- **36-second open loop.** Repeat a deterministic hash order over all queries in each existing CoF panel, at 12 requests per second. Each route issues the same 432 requests and completes every issued request. Report P95 measured from scheduled arrival to full response for the full trace and each successive 12-second arrival window. Count completions by wall-clock window separately. The arrival schedule continues through the full trace even when requests queue. Drained throughput includes the time after the last arrival.
- **18-second closed loop.** Each of four workers immediately submits its next request when it finishes one, cycling through the same query panel. Count completions strictly within the measurement interval and compare every response, including drain, with the padded references from the open-loop trace. Closed-loop throughput measures this workload at four active requests; it is not a general service capacity guarantee.

The two datasets reverse the open-loop execution order, then reverse it again for the closed-loop phase. Each environment is warmed before measurement. There is no HTTP transport or empirical serving background workload. Request outputs are compared as ordered experts and ordered citations; released labels do not independently validate citation correctness.

## Results

Slurm job 42824 ran both panels on one RTX 5090. The packed path shares 1,191,553,024 bytes of read-only parameters, with zero duplicated parameter bytes. All 864 packed open-loop responses and all 547 packed closed-loop responses reproduce the padded experts and ordered citations.

| Panel | Route | Open-loop P95 | Open-loop P95 by arrival window | Wall-clock completions by window | Closed-loop completions/s |
| --- | --- | ---: | ---: | ---: | ---: |
| KDD (122 unique queries) | Padded four | 1,558.51 ms | 696 / 1,150 / 1,646 ms | 137 / 139 / 137 | 11.28 |
| KDD | Packed eight | 167.19 ms | 138 / 182 / 129 ms | 144 / 143 / 145 | 14.94 |
| SciRepEval (59 unique queries) | Padded four | 1,354.90 ms | 578 / 966 / 1,460 ms | 138 / 139 / 138 | 11.56 |
| SciRepEval | Packed eight | 124.33 ms | 124 / 124 / 124 ms | 144 / 144 / 144 | 15.00 |

At 12 requests/s, padded latency rises across the observed trace and replies accumulate in the queue. Packed open-loop drained throughput is 12.02 and 12.01 requests/s, compared with 11.53 and 11.61 for padded. The closed-loop throughput tests suggest a capacity difference at four active workers on this GPU. They increase completed requests/s by 32.5% and 29.8%, with P95 of 427.62 and 407.59 ms under the closed-loop backlog. The large open-loop latency reduction includes queueing. The service timings do not establish capacity for arbitrary workload mixes or independent evidence quality.

[KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json)

## Reproduce

Prepare the verified checkpoints and public CoF snapshots from the [concurrent experiment](../cof-ragged-concurrent-v1), then run on a single Slurm CUDA compute node.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_ragged_sustained \
  KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_KDD_RESULTS
PYTHONPATH=src:. python -m experiments.retrieval.cof_ragged_sustained \
  SciRepEval EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_SCIREPEVAL_RESULTS
```
