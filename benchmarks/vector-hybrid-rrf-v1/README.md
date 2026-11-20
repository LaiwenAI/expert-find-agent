# Vectorize exact reciprocal-rank fusion for hybrid source retrieval

Preserve the original exact ranking contract. Both BM25 and dense
ranks break score ties by integer document position. The fused score
is the sum of reciprocal ranks with constant 60. Every active BM25
row and every eligible dense row contributes. Each method returns
the first 128 fused positions in the same order.

Compare a scalar Python implementation aligned with the
[live ranker](../live-joint-source-request-v1), full vectorized sorting,
and vectorized fusion with partial selection of the final top 128.
The array versions map the resident BM25 row order to dense document
positions, calculate ranks using score and position as keys, and
accumulate reciprocal contributions in float64. Final partial
selection keeps every position above the cutoff, chooses the smallest
positions among boundary ties, then orders the retained positions.
This changes the implementation while preserving ranking semantics.

Two evaluations measure separate costs:

- **Live publisher search.** Use the 452-article
  [mixed catalog](../mixed-publisher-joint-v1) and the original
  211 PLOS/eLife requests. For every arm encode both topics live,
  compute exact dense scores, calculate BM25 and fusion, nominate
  by complete personal-role postings and open three local source
  declarations. Compare each ranked list with the original live
  entry point and check every serialized expert and citation
  byte-for-byte. Request timing runs from JSON parsing through
  response encoding. Component timing starts after dense scores
  are available and includes BM25, both rank orders and fusion.
- **Large public component test.** Use the frozen LitSearch title
  and abstract projection, its existing dense-vector cache and the
  original manually authored queries. Use the same score arrays
  in every arm. Time BM25, dense ordering and fusion on
  those arrays. This isolates ranking from query encoding,
  dense scoring, user-facing transport and evidence verification.
  It applies the full-list RRF contract consistently, independently
  of the previously published truncated-list LitSearch configuration.

One thousand additional generated cases check exact agreement with
sparse row permutations, absent sparse matches, excluded documents
and ties at several positions.

## Results on RTX 5090

Both tasks of Slurm array `44642` on ws5 completed. Every ordered
list agrees with the scalar reference. The live expert replay
also exactly preserves every response, including 85 known PLOS
and 46 known eLife matches.

| Scope | Scalar Python P95 | Vectorized full sort P95 | Vectorized partial selection P95 |
| --- | ---: | ---: | ---: |
| 452 articles, BM25 and fusion component | 0.84–0.87 ms | 0.147–0.149 ms | 0.154–0.157 ms |
| 452 articles, complete warm request | 10.55–10.78 ms | 10.03–10.26 ms | 10.05–10.26 ms |
| 64,183 LitSearch articles, BM25 and fusion component | 79.69–80.75 ms | 10.92–10.95 ms | 7.82–7.89 ms |

The large test covers all 246 manually authored LitSearch queries.
At that size, partial selection also improves on the vectorized
full-sort control. The 452-article test favors the simpler full
sort within the ranking component, and the end-to-end improvement
is about 0.52 ms. The large test measures a source-ranking kernel,
with no expert-ranking quality or production latency estimate.
Both dense and sparse rank lists are still computed in full
before selecting the fused prefix. This experiment leaves
open how to avoid scoring all documents at larger scale.

## Reproduce

Restore the pinned publisher XML inventories, model and LitSearch
projection and vector cache. The [existing LitSearch benchmark](../litsearch-adaptive-depth-v1)
documents the original public inputs and embedding protocol.
Run in a Slurm GPU allocation with a fresh output path and repeat
with `--order-offset 1`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.vector_hybrid_rrf \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json \
  LITSEARCH_PROJECTION LITSEARCH_VECTORS NEW_RESULTS
```

[Verification counts and timings](results.json).
