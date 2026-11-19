# Live global lexical and dense fusion on CoF KDD

Use the 122-query slice from the [global candidate audit](../cof-global-candidate-audit-v1/README.md). Every arm searches all 737 candidate reviewers through the same 99,594 owned sources. No query-specific judged pool is supplied. Each arm returns ten reviewer candidates and attaches up to two model-supported sources to each of its first five reviewers, without moving them. The annotation stage uses the same frozen verifier and shared frontier policy.

| Arm | Candidate and source ranking |
| --- | --- |
| Dense | Maximum owned-source embedding cosine, with citations drawn from its top three sources |
| BM25 | Mean of the top three owned-source BM25 scores, zero padded to three |
| Source fusion | Add dense and BM25 reciprocal ranks per source, then take the best owned source |
| Expert fusion | Add reciprocal ranks of dense and lexical expert aggregations |
| Reserved union | Reserve five reviewers per channel before deduplication and fill; order the selected set by expert fusion |

Expert fusion and reserved union use the same fused source scores to select evidence. RRF uses a fixed constant of 60, competition ranks for ties, and zero contribution for unmatched sparse sources. The two fused indexes are loaded once. Timing includes live query encoding when required, retrieval, and the same evidence attachment stage. It excludes networking and index initialization.

Reciprocal rank fusion is an established baseline. [Cormack et al., SIGIR 2009](https://doi.org/10.1145/1571941.1572114)

CoF's labels cover a sparse pool of reviewers. Recall and P@5 measure recovery of known positives. Source scores and citation coverage do not establish the semantic correctness of the citations.

## Results on one RTX 5090

| Arm | Known-positive Recall@10 | Known-positive P@5 | Positive and cited P@5 | Retrieval P95 | Complete P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 13.64% | 3.77% | 2.13% | 9.2 ms | 123.8 ms |
| BM25 | 14.05% | 4.10% | 1.31% | 20.8 ms | 143.9 ms |
| Source fusion | 14.88% | 3.44% | 1.80% | 44.8 ms | 160.3 ms |
| Expert fusion | 14.46% | 4.26% | 1.80% | 44.9 ms | 170.2 ms |
| Reserved union | 16.94% | 4.43% | 1.80% | 45.0 ms | 170.4 ms |

Relative to dense, the reserved union captures twelve additional known-positive reviewer-query pairs and loses four. Its paired P@5 difference is +0.66 percentage points (95% bootstrap interval −0.49 to +1.80). It yields a higher known-positive recall point estimate and lower positive-and-cited P@5, with 46.6 ms higher complete P95. Only 13.61–16.23% of the returned slots across arms receive labels in the original pool. A stronger ranking claim needs additional judgments or another dataset with better coverage. [Aggregates](results.json)

## Reproduce

Use the same checkpoint and source-vector snapshot as the linked global audit. Build or retrieve the compiled CoF KDD source-level BM25 index using the existing CoF lexical pipeline, and verify its hash against the results. The source IDs must match the dense inventory.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_global_hybrid \
  EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT FAST_SQLITE NEW_RESULTS
```
