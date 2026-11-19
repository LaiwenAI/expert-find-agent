# Defer source fusion until reviewers are shortlisted

Reserved union ranks reviewers using their best dense source and zero-padded mean of their three strongest BM25 sources. It reserves the leading five reviewers in each channel, deduplicates, and fills to ten using expert reciprocal-rank fusion. The [eager implementation](../cof-global-hybrid-v1/README.md) also constructs and sorts source fusion scores across all reviewer profiles before it knows which profiles it needs.

The deferred implementation preserves the same candidate selection rule. It aggregates positive BM25 contributions for the three-source mean, with a full-sort fallback for negative weights. After forming the ten-reviewer shortlist, it collects their owned sources and computes the dense and sparse competition ranks only at those source rows. Sorted global score arrays give exact ranks by right-sided binary search, including ties. Unmatched sparse sources contribute zero to source fusion and remain in the global rank distribution. Each shortlist reviewer keeps the same three-source ranking with source order as the tie-break. Verification is unchanged.

This reorders the calculation of an existing ranking and citation policy. A 1,800-case randomized local check matched the eager and deferred ranked expert-source lists, including co-owned sources, ties, sparse nonmatches, negative weights and fewer than ten eligible owners.

## Live CoF KDD comparison on RTX 5090

The 122-query panel, 737 reviewers, 99,594 source vectors, BM25 index, open Qwen checkpoints, and labels match the prior experiment by hash. Each arm runs live query encoding, retrieval, and citation attachment with alternating execution order. Weights and indexes are resident. Times exclude transport and model or index initialization.

| Route | Known-positive Recall@10 | Known-positive P@5 | Positive and cited P@5 | Retrieval P95 | End-to-end P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 13.64% | 3.77% | 2.13% | 10.3 ms | 125.9 ms |
| Eager reserved union | 16.94% | 4.43% | 1.80% | 46.8 ms | 172.7 ms |
| Deferred reserved union | 16.94% | 4.43% | 1.80% | 32.4 ms | 157.9 ms |

All 122 requests matched between the eager and deferred arms on ordered candidates, source shortlists, and final citations. Retrieval P95 was 30.8% lower and end-to-end P95 was 8.6% lower than the eager union in this run. Dense alone remains faster and has higher positive-and-cited P@5. The union's known-positive P@5 gain is 0.66 percentage points over dense (paired 95% bootstrap interval −0.49 to +1.80). CoF labels are incomplete and cover reviewers, not their individual citations. These results measure exact output preservation for the fusion rearrangement and benchmark-relative recovery of known positives. [Aggregate results](results.json)

## Reproduce

Use the same datasets, checkpoints, and lexical index as the linked eager benchmark. Run on a Slurm GPU compute node:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_global_hybrid \
  EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT FAST_SQLITE NEW_RESULTS --deferred-union
```
