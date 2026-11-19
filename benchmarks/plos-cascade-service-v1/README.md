# Measure queueing and GPU batching in the frozen attribution reader

Use the 45 [independent legacy contribution-note cases](../plos-independent-cascade-v1/README.md) and the frozen character gate and reader cutoff from its [development run](../plos-legacy-note-cascade-v1/results.json). Leave all evidence text, queries and decisions unchanged. One request verifies a pair of candidates for two duties against one already located publisher note.

Compare four in-process routes. Cross always reading versus routing with the frozen character gate, and one request pair per model batch versus up to eight. A producer releases a common Poisson arrival trace independently of GPU worker progress. When the character gate settles both candidates, it responds without joining the GPU queue. Otherwise the GPU sees both candidates for consistent input construction. The batch worker combines pending pairs, waits up to two milliseconds after the first ready request, and dispatches accrued backlog immediately. It rechecks a pair singly when a pending margin lies within ±0.25 of the reader cutoff. The guard is a fixed test setting. Verify actual output parity against the corresponding single-pair route.

Nominal rates are 20, 55 and 90 requests/s. Use three shared seeded traces per rate, two randomized passes over the 45 papers per trace, with rotated arm order and warmed reader shapes. Each trace has 90 requests. The release includes realized scheduled rate, completion throughput, dispatch-queue wait, arrival producer lag, latency quantiles, model forwards, rechecks and outcomes. The summary takes the median of the three trial statistics. Latency spans scheduled arrival to completion, including preprocessing and queueing. It excludes source discovery and network transport. The finite traces reuse publications and therefore do not create additional independent quality judgments.

## RTX 5090 results

Job 42641 completes 36 traces, with 90 requests in each. The median scheduled rates are 19.34, 53.18 and 87.02 requests/s. Producer-lag P95 stays below 0.1 ms across the median traces. The table shows request P95 in milliseconds and realized completions per second. Parenthesized throughput is measured over the finite trace and includes draining its queue.

| Route | Low load, P95 (req/s) | Medium load, P95 (req/s) | High load, P95 (req/s) |
| --- | ---: | ---: | ---: |
| Single-pair reader | 49.90 (19.24) | 353.60 (44.70) | 943.87 (44.38) |
| Single-pair character gate | 36.38 (19.34) | 71.35 (53.05) | 269.67 (70.40) |
| Batched reader | 62.85 (19.24) | 270.54 (50.50) | 836.00 (50.44) |
| Batched character gate | 40.15 (19.34) | 76.32 (52.99) | 259.15 (78.26) |

The character gate reduces GPU-read pairs from 90 to 56 on every trace. The low and medium loads favor its single-pair variant in these finite trials. At the highest load, batching its fallbacks increases observed completion throughput from 70.40 to 78.26 requests/s and lowers median P95 from 269.67 to 259.15 ms. Both still fall behind the arrival trace, so these are overload measurements. At the medium load, the single-pair gate nearly keeps up while the unfiltered single-pair reader queues, which accounts for most of the large P95 difference.

When batched, the character gate reproduces all 810 corresponding single-pair request decisions across nine traces. The unfiltered batched reader agrees on 86–87 of 90 decisions in each medium- and high-load trace, despite the near-threshold guard. Its changing numerical decisions prevent an exact-output claim. For both character routes, a trace returns 80 supported authors and 14 wrong authors, while a single-pair reader returns 66 and 16. These counts repeat the 45 underlying labels twice.

Gate decisions and batching must be evaluated separately. Skipping calls changes which work reaches the GPU and gives the largest measured queueing benefit in this workload. Batching yields a smaller throughput benefit at the highest load and requires independent output checks. This result measures a supplied-note attribution component in-process. It does not report service latency for the retrieval agent. [Aggregate results](results.json)

## Reproduce

Restore the two captured publisher snapshots and use the same frozen Qwen3-4B checkpoint as [the independent study](../plos-independent-cascade-v1/README.md). Run inside a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.legacy_cascade_service \
    OLD_SNAPSHOT INDEPENDENT_SNAPSHOT QWEN4B TRANSFER_RESULTS CASCADE_RESULTS NEW_RESULTS
```
