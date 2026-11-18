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

## Follow-up: cheaper confirmation

Slurm job 40540 repeated the same split and compared three confirmation strategies. Each threshold was chosen on the original 11 calibration clues. The LitSearch head retained all seven coefficients learned on 128 separate LitSearch questions. Its input uses a four-candidate shortlist here, compared with six in training.

| Method | Correct source confirmed | Wrong source, target withheld | Present-view P95 |
| --- | ---: | ---: | ---: |
| Dense score and gap | 9/24 | 0/24 | 7.10 ms |
| BM25 and dense top hit agree | 16/24 | 1/24 | 7.22 ms |
| Frozen LitSearch head | 5/24 | 0/24 | 7.17 ms |
| Verify dense top hit using one cross-encoder call and dense gap | 18/24 | 0/24 | 14.94 ms |
| Four-candidate reranking | 17/24 | 0/24 | 33.13 ms |

All five methods made zero wrong confirmations in the present view. Top-hit verification uses its absolute cross-encoder score and the gap between the first two dense results. The one-candidate route had 55% lower observed P95 than four-candidate reranking on this reused diagnostic. The sample provides limited evidence about the false-confirmation rate, and top-hit verification cannot recover a misranked source. [Follow-up results](gates-results.json)

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

To reproduce the follow-up, append `--transfer-head benchmarks/plos-credit-source-paraphrases-v1/litsearch-head.json` and use a new output path. The head's training protocol is in [the LitSearch experiment](../../experiments/litsearch/learned_confirmation.py).
