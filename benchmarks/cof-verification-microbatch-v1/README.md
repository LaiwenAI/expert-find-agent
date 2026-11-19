# Evidence verification with cross-request GPU batching

The benchmark draws 48 queries by a fixed hash from the eligible CoF KDD panel. It excludes query IDs and titles overlapping reviewer profiles and titles present in the SIGIR, NIPS and SciRepEval query panels. KDD has 170 eligible queries under these rules. The benchmark uses a warm resident encoder and exact candidate index, a frozen Qwen3-Reranker-0.6B, and two required sources per confirmed expert. The reranker sees the query and source, so every policy can reuse a score within one request. Candidate ownership remains a separate check.

Compare immediate waves, fixed length-sorted batches of eight, and deferred frontier verification. The one-request configuration uses the previous [HTTP runner](../cof-verification-http-v1/README.md). The multi-request configuration admits up to four request workers and serializes physical work on the same GPU. It combines ready dynamic scorer inputs in the same 128-token length bucket for up to 1 ms, with at most 16 rows per launch. It preserves per-request query encoding and fixed-shape rechecks. Each policy uses the same batch broker and 0.25 guard. Scores are never reused across requests.

The open-loop client uses two matched orders at each offered rate. Time runs from scheduled arrival through response completion on persistent loopback connections. The service time for a concurrent request includes waits inside the GPU broker. Bytes are compared with a fresh exhaustive fixed-shape result for the same query. Timing reflects finite warm traces on one GPU. The task judges reviewer relevance. The exhaustive comparison verifies output agreement under the selected model and ranking rule.

Iteration-level scheduling and selective batching have established precedents in model serving. This benchmark uses request interleaving to test the cost of a rank-and-citation stopping policy under a common GPU scheduler. [Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)

## Results on RTX 5090

| Offered requests/s | Route | One worker P95 | Four workers P95 | Four workers drained requests/s | Merged GPU launches |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | Immediate waves | 254 ms | 275 ms | 4.03 | 0 |
| 4 | Length buckets | 240 ms | 242 ms | 4.03 | 0 |
| 4 | Deferred frontier | 225 ms | 235 ms | 4.03 | 0 |
| 10 | Immediate waves | 3,378 ms | 3,613 ms | 5.80 | 155 |
| 10 | Length buckets | 4,168 ms | 4,588 ms | 5.13 | 71 |
| 10 | Deferred frontier | 2,966 ms | 3,079 ms | 6.20 | 170 |

The 1,152 HTTP responses from job 40679 match the exhaustive reference. At 10 requests/s and four workers, the deferred route reduces P95 by 14.8% relative to the fastest fixed route with the same broker. With one worker, P95 is lower for both of those policies than with four workers. Within the deferred route the added broker raises P95 from 2.97 to 3.08 seconds and changes drained throughput from 6.18 to 6.20 requests/s. Only seven of 96 deferred replies in the four-worker arm meet the 500-ms deadline at 10 requests/s. At 4 requests/s, the observed GPU launches never merge requests and all 96 replies per arm meet the deadline. [Aggregates and broker counters](results.json)

## Reproduce

On a Slurm CUDA compute node with the pinned public CoF inputs and checkpoints, run:

    PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1       python -m experiments.retrieval.cof_verification_microbatch       COF_INPUT_MANIFEST EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON

The public artifact contains aggregate metrics, selected query IDs and raw-row provenance.
