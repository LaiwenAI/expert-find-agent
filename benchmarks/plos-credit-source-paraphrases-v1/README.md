# Unquoted paper clues and missing-source controls

This diagnostic tests whether a system can confirm a particular paper from an unquoted clue and abstain when that paper is unavailable. It uses 24 test DOIs from the external PLOS CRediT fixture and 11 different DOIs for threshold calibration. There is one English clue per DOI. The assistant wrote clues from public titles and abstracts before running retrieval. The input check rejects a copied eight-word span. Each clue is replayed with its target available and with that DOI masked before shortlist selection.

BM25, the existing ordered-match guard, frozen Qwen3-Embedding-0.6B, and Qwen3-Reranker-0.6B use the same 165-paper title-and-abstract catalog. The reranker scores the four dense candidates. Dense and reranker thresholds maximize correct confirmations with zero wrong confirmations in either calibration view. The ordered guard retains its excerpt thresholds. Lexical corpus statistics remain fixed in the masked view.

## Results

Slurm job 40534 completed on an RTX 5090. Each view has 24 test clues.

| Method | Target in shortlist | Correct source confirmed | Wrong source, target present | Wrong source, target withheld | Present-view P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 top hit | 21 | 18 | 6 | 24 | 0.15 ms |
| BM25 with ordered guard | 21 | 0 | 0 | 0 | 0.57 ms |
| Dense with score and gap guard | 24 | 9 | 0 | 0 | 7.17 ms |
| Dense with four-candidate reranking and guard | 24 | 17 | 0 | 0 | 33.14 ms |

Dense retrieval ranked 23 of the 24 targets first before confirmation. The reranker ranked all 24 first, then abstained on seven. A post hoc replay that accepted the dense result when its frozen guard passed and otherwise used the reranked result confirmed 19 targets with zero wrong selections in both views. It invoked reranking for 15 present and all 24 withheld clues. The replay has no measured cascade latency.

This small, model-authored set tests source identification. Independent human requests, larger catalogs, contributor attribution and tail-latency effects need separate tests.

## Reproduction

Build the catalog using the `prepare` command documented in [the external benchmark](../plos-credit-external-search/README.md). Run on a Slurm-allocated CUDA node with the verified local checkpoints.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.credit.source_paraphrases \
  artifacts/plos-credit-external-search-catalog-v1.json \
  benchmarks/plos-credit-external-v1/manifest.json \
  benchmarks/plos-credit-source-paraphrases-v1/queries.json \
  artifacts/models/qwen3-embedding-0.6b \
  artifacts/models/qwen3-reranker-0.6b \
  artifacts/plos-credit-source-paraphrases-v1.json
```

The generated JSON includes source ranks, score gaps, thresholds, per-request timing and checkpoint hashes. [Published results](results.json) include aggregate counts, thresholds and provenance. P95 covers warm local source retrieval and confirmation scoring. It excludes offline indexing, model load, author attribution, transport and queueing.
