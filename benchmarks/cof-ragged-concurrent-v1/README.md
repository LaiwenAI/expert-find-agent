# Packed evidence verification under concurrent arrivals

The [sequential benchmark](../cof-ragged-frontier-v1) measured lower verifier latency with unpadded attention, using a prototype that switched the model's backend during execution. This experiment uses a separate packed backbone configured once before accepting requests. Ordinary four-row rechecks stay on the existing canonical model. Report the extra resident weights required by this implementation. Each driver uses the same memory layout, including both verifier paths.

Four CPU workers accept a matched open-loop trace, and a shared broker serializes the encoder, retrieval and verifier work on one GPU. With a fixed seed, sample 40 requests from each previous public cohort, then replay two orders for each offered rate and route. All requests use live query encoding and full-inventory top-five retrieval. The three routes have the same evidence rule, shared-source score reuse and fixed recheck dimensions:

| Route | Dynamic verification |
| --- | --- |
| Padded four | At most four per-request rows with ordinary dynamic padding |
| Packed eight | At most eight per-request rows with independent variable-length causal attention |
| Packed eight + merge | Same per-request limit; the broker can combine ready dynamic batches across requests for up to 1 ms, with a maximum of 16 rows per launch |

A merged launch keeps separate attention offsets for every document. No scores pass from one request to another. Fixed rechecks and retrieval are not merged. The broker is shared, with dedicated CPU threads forming prompts and making frontier decisions. The benchmark uses direct in-process calls and no HTTP transport. Timer starts at the scheduled arrival and includes CPU worker and GPU waiting. Drained throughput includes the tail after the last arrival. Each complete ordered response is compared with a fresh padded-four response on the same trace. The full padded response is additionally checked for consistency across all repeated arrivals.

Offered rates are 4, 8 and 12 requests per second. This study measures a finite warm workload, scoped to the selected public reviewer queries and one GPU. It reports output agreement, independent of whether the citations are judged correct by a person. Hydration, model startup, streaming, network transit and production background work are outside the measured times.

## Results

Slurm job 42808 ran the matched traces on one RTX 5090. The two packed routes reproduce all 960 ordered responses and citations relative to padded verification. Each table cell is arrival-to-response P95; the right column gives drained throughput for padded / packed / packed with merging.

| Panel | Offered requests/s | Padded four | Packed eight | Packed eight + merge | Drained requests/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| KDD | 4 | 126.57 ms | 92.81 ms | 96.55 ms | 4.07 / 4.09 / 4.09 |
| KDD | 8 | 128.64 ms | 94.68 ms | 99.72 ms | 8.06 / 8.14 / 8.14 |
| KDD | 12 | 463.54 ms | 108.51 ms | 113.90 ms | 11.43 / 12.16 / 12.15 |
| SciRepEval | 4 | 120.03 ms | 97.71 ms | 101.07 ms | 4.06 / 4.06 / 4.06 |
| SciRepEval | 8 | 124.82 ms | 98.54 ms | 103.12 ms | 8.02 / 8.05 / 8.04 |
| SciRepEval | 12 | 353.74 ms | 112.89 ms | 120.58 ms | 11.48 / 11.95 / 11.93 |

Packed per-request verification lowers low-load P95 by 18.6–26.7%. At 12 requests/s, the measured finite-trace P95 falls by 76.6% and 68.1%. In that setting, P95 waiting for a CPU worker falls from 95.92 to 0.12 ms on KDD and from 8.60 to 0.10 ms on SciRepEval. The larger loaded gain includes relief from queueing and should not be interpreted as a per-forward speedup or a steady-state capacity estimate. Cross-request coalescing occurs only once and raises observed P95 in all six conditions. The per-request packed route is the stronger default in this workload.

The multi-rate runner creates a separate 1,191,553,024-byte packed parameter copy. The following audit tests whether that storage can be shared.

[KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json)

## Parameter storage check

A separate runner builds two fixed packed-backend views. One duplicates backbone parameters and the other aliases the original read-only parameters while keeping module configuration and buffers independent. It compares every parameter pointer and replays the identical 12-request/s sequence through both views and the padded reference. This is a response and parameter-storage audit, separate from the multi-rate timings above. Job 42811 shares all 1,191,553,024 parameter bytes with zero duplicated parameter bytes. All 40 KDD responses match both the copied packed model and the padded reference at 12 requests/s. Separate configuration and buffers remain allocated. A forced two-group broker probe coalesces both groups in one launch and returns each score to its caller, with a maximum absolute difference of 0.0234375 logits versus standalone packed scoring. The open-loop shared-weight replay has no naturally merged batches. [Storage and merge audit](kdd-shared.json)

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_ragged_shared_weights \
  KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_STORAGE_RESULTS
```

## Reproduce

Use the same public inputs and checkpoints as the [sequential benchmark](../cof-ragged-frontier-v1), with a CUDA device that supports PyTorch variable-length attention.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_ragged_concurrent \
  KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_KDD_RESULTS
PYTHONPATH=src:. python -m experiments.retrieval.cof_ragged_concurrent \
  SciRepEval EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_SCIREPEVAL_RESULTS
```
