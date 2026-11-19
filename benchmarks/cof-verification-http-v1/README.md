# Evidence verification under offered HTTP load

Slurm job 40663 uses one RTX 5090 and a resident query encoder, exact owned-source index, and cross-encoder verifier. Reuse the 59 title-disjoint SciRepEval queries from the [length-bucket comparison](../cof-live-certified-batching/#length-bucketed-fixed-batches-and-shared-scores). Each request requires two supporting sources. The three service arms are immediate rechecking with a request-local cache, fixed length-bucketed scoring with groups of eight and immediate rechecking, and deferred frontier verification with a request-local cache. A fresh exhaustive fixed-shape reference is produced in the same allocation.

A client schedules all 59 requests independent of completions at each offered rate. Two seeded orders per rate are identical across policies. The worker serves one request at a time. The timer runs from scheduled arrival through HTTP response completion on persistent loopback connections. It includes queueing, fresh query encoding, retrieval, prompt construction, verifier calls and result serialization. Model initialization, warmup and reference generation happen before timing. Scoring caches remain within a request.

| Offered requests/s | Route | Client P95 | Server queue P95 | Drained requests/s | Completed within 500 ms / 118 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | Immediate waves | 252 ms | 0.74 ms | 4.02 | 118 |
| 4 | Length buckets | 230 ms | 0.06 ms | 4.03 | 118 |
| 4 | Deferred frontier | 201 ms | 0.08 ms | 4.02 | 118 |
| 8 | Immediate waves | 2,833 ms | 2,643 ms | 5.76 | 14 |
| 8 | Length buckets | 3,137 ms | 2,956 ms | 5.57 | 14 |
| 8 | Deferred frontier | 1,758 ms | 1,587 ms | 6.47 | 20 |

All 708 responses match the corresponding reference bytes and therefore the ordered experts and cited source IDs. Against the fastest measured fixed route at each load, deferred verification lowers P95 by 12.7% at four requests/s and 38.0% at eight requests/s. At eight requests/s its drained throughput rises by 12.3%, but backlog persists and most requests miss 500 ms. Both repeats show the direction of the P95 gain. Throughput is completed requests divided by the full trace and drain time. These single-worker finite traces exclude cross-request microbatching, external network costs and independent semantic labels for the citations.

The Slurm transport snapshot differs from the release transport only in the unused earlier retrieval-only batching helper. The executed HTTP server and client code have identical Python syntax trees. The report pins both versions. [Aggregates](results.json)

## Reproduce

On a Slurm CUDA compute node with the same verified CoF data and checkpoints, run:

    PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python -m experiments.retrieval.cof_verification_http \
      COF_INPUT_MANIFEST EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON

The public report excludes individual request rows and records their hash.
