# Shared optimistic citation checks on CoF KDD

Use the 122 title-disjoint KDD queries remaining after excluding the 48 requests in the prior [HTTP microbatch study](../cof-verification-microbatch-v1/README.md). The larger KDD panel has appeared in earlier studies. This comparison uses published reviewer labels, a resident exact source index, a live frozen query encoder, and a live frozen cross-encoder. Each request retrieves ten candidate reviewers and three articles per reviewer. A reviewer is promoted with two positive article judgments, and canonical citations follow the original article order.

The existing deferred route advances source cursors for eligible reviewers. The shared frontier route examines only unresolved citations in the optimistic Top-5 response. Each query-document relevance verdict is shared among confirmed co-owners. The cost-aware variant orders each wave by the sum of reciprocal optimistic ranks helped by a source divided by its token count. The shared and cost-aware variants keep the same 0.25-logit guard and fixed-shape rechecks as the existing deferred route. Outside-band decisions rely on the empirical numerical guard. Sharing applies to query-document relevance; author-specific contribution judgments require their own verification.

Four existing controls run alongside the two new arms. Arm order rotates per request, and each arm performs separate live query encoding, retrieval, tokenization, verification and ranking. Report serial warm-request latency, excluding networking and loaded-index initialization. Synthetic exhaustive auditing checked both schedulers over 9,600 random cases with shared sources, under stable verdicts outside the guard.

## RTX 5090 results

| Route | P50 | P95 | Mean GPU calls | Exact expert order and citations / 122 |
| --- | ---: | ---: | ---: | ---: |
| Fixed-shape exhaustive | 289.8 ms | 383.8 ms | 8.00 | 122 |
| Guarded immediate | 157.9 ms | 246.4 ms | 6.82 | 122 |
| Guarded length buckets | 192.3 ms | 257.5 ms | 4.10 | 122 |
| Existing deferred | 151.1 ms | 219.3 ms | 6.70 | 122 |
| Shared frontier | 146.8 ms | 217.3 ms | 5.63 | 122 |
| Cost-aware frontier | 153.2 ms | 215.0 ms | 5.69 | 122 |

Against the existing deferred route, the shared frontier saved 7.95 ms per query on average (paired bootstrap 95% interval 4.01 to 12.13 ms), and the cost-aware variant had the lowest P95 with a 1.94% reduction. The latter used slightly more scored documents and GPU calls than the unordered frontier. All routes reached 3.77% hard-label P@5 and 2.62% P@5 when also requiring returned citations. The benchmark has labels for reviewer relevance; citation agreement is against the fixed-shape model output. These data isolate scheduling behavior and expose weak overall reviewer matching in this configuration. [Aggregate results](results.json)

## Reproduce

Inside a Slurm GPU allocation with the pinned CoF dataset and vector snapshot:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_shared_frontier \
  UNUSED_SIGIR_MANIFEST EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```

The SIGIR manifest argument is retained by the shared runner interface. CoF KDD uses its own hashed data and model inputs.
