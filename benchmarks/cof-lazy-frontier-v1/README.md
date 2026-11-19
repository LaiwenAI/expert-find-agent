# Build reranker prompts when the evidence frontier needs them

Use the same 122 KDD queries, frozen reviewer candidate generator, two-citation rule, and 0.25-logit guard from the [shared frontier study](../cof-shared-frontier-v1/README.md). Compare the existing deferred and shared frontier routes with a shared frontier that tokenizes a query–source prompt only when it is first scored. Reuse token IDs within the request. If a fixed-shape recheck is needed, finish tokenizing the original pool before choosing its padding width. This preserves the reference width and avoids changes to the shared frontier scheduler or guard policy.

Each arm independently executes live query encoding, candidate retrieval, tokenization, verification and ranking. Arm order rotates across requests. Check exact equality of the candidate pool, scored margins, ranked experts, citations, number of forwards and padded tokens for eager versus lazy shared frontier. Report warm sequential processing, excluding index initialization, networking and request queueing. Reviewer relevance uses the published hard labels. Counting a returned citation does not certify human-judged source support.

## Results on RTX 5090

| Route | Mean prompts tokenized | Mean GPU calls | Mean latency | P95 | Exact response vs. eager |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing deferred | 23.57 | 6.70 | 152.73 ms | 218.39 ms | 122/122 |
| Eager shared frontier | 23.57 | 5.63 | 144.86 ms | 214.11 ms | 122/122 |
| Lazy shared frontier | 18.60 | 5.63 | 143.81 ms | 214.89 ms | 122/122 |

Job 41815 replicated the shared frontier's workload savings and returned identical score margins, experts and citations with lazy prompt construction. It tokenized 21.1% fewer prompts and saved 1.05 ms on average versus eager tokenization (paired query-bootstrap 95% interval 0.60 to 1.52 ms). P95 was 0.78 ms higher in this run. All arms have hard reviewer P@5 of 3.77%, and 2.62% when also requiring a returned citation. The experiment supports demand-driven tokenization as a small execution optimization. It does not provide evidence of a material P95 improvement or better matching. [Aggregate results](results.json)

Run with the same snapshots as the shared frontier study inside a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.cof_lazy_frontier \
  UNUSED_SIGIR_MANIFEST EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```
