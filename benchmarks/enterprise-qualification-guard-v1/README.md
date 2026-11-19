# Qualification-boundary rechecking for first-query expert search

The search route sends only prior-qualification clauses to evidence verification. The generic recheck guard compares the two largest role logits, including ties between roles that both leave verification unchanged. The qualification guard compares the prior-qualification logit with the maximum across all other roles. It rechecks the original full-context block if any clause's absolute gap is at or below 2.0. The cutoff, model, cached pass, and fallback batch geometry are shared with the generic guard.

On identical logits, the qualification gap is at least the top-two gap. The qualification guard can therefore skip full-block rechecks. The cutoff is empirical. A qualification gap strictly larger than twice a valid bound on absolute logit drift would establish conditional stability. This experiment measures agreement and does not establish such a drift bound.

## End-to-end comparison on one RTX 5090

Use the same 80 enterprise development requests, 52,667 indexed sources, and open Qwen models as the [prior first-query benchmark](../enterprise-first-query-v1/README.md). Confirm both input digests. Run each arm from raw request to ranked and cited output. Across the four arms, balance order position and first-order carryover. The fixed pipeline uses live query encoding, exact source scoring, owned-source lookup, and evidence verification. The time window starts at request parsing and ends at ranking. Weight loading, offline corpus indexing, and service transport sit outside the window.

| Reader route | P50 | P95 | Rechecked blocks | Changed qualification sets / 80 | Changed expert order / 80 | Changed citation or unresolved state / 80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent full-context batches | 571 ms | 5,476 ms | 0 | 0 | 0 | 0 |
| Prefix cache | 192 ms | 643 ms | 0 | 3 | 0 | 3 |
| Generic top-two guard | 231 ms | 3,425 ms | 56 | 0 | 0 | 0 |
| Qualification guard | 195 ms | 2,153 ms | 25 | 0 | 0 | 0 |

The qualification guard used 55.4% fewer rechecks and had a 37.2% lower P95 than the generic guard in this run. It changed ten four-way clause labels across 1,452 clauses relative to the independent arm, while preserving all qualification sets. The paired request bootstrap gives a P95 speedup interval of 1.69–5.99× against independent full-context batches (2.54× point estimate). The reported 37.2% decrease against the generic guard is a point comparison.

The unchanged outputs establish agreement with a frozen model on reused development cases. Independent adjudication and sealed-cohort replication are pending. The same test also reproduces three output-state changes on the cache-only arm despite unchanged expert order, so use the citation and unresolved-state checks alongside ranking agreement.

## Reproduce

Use the development-only projections and checkpoint digests in the [prior benchmark](../enterprise-first-query-v1/README.md). Run in a Slurm CUDA allocation:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_first_query \
  --queries DEVELOPMENT_QUERIES --corpus DEVELOPMENT_SOURCES \
  --models CHECKPOINT_DIRECTORY --output NEW_AGGREGATE --qualification-guard
```

The [aggregate results](results.json) record Slurm job 41115, model and input hashes, paired timing against the independent arm, and output agreement. The release does not include requests, source text, or case-level decisions.
