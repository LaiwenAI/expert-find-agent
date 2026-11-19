# Low-cost confirmation of expanded publication pairs

Use the unchanged [PLOS source-confirmation panel](../plos-source-confirmation-v1/README.md) of 25 linked paper pairs in 15 components. Search the 171-paper index with the frozen Qwen3-Embedding-0.6B model and BM25. Use result excerpts and keyword clues with both required sources present and with either source masked. First verify aggregate agreement with the original top-four and expanded rankings.

Each selective gate keeps any top-four answer. If top-four abstains and the rank-12 join supplies a pair, the gate tests that proposal. Compare the worse source's dense rank, full-clue embedding similarity, and IDF-weighted lexical clue coverage. The two fragment gates split each source clue into contiguous word halves. With one extra batched encoder forward, score four fragments against the index. Use the weakest fragment similarity, or the weakest gap between each proposed source and its strongest other visible document. The candidate pair is fixed while the gates score it.

Calibrate each score independently by taking the largest false-proposal score on all components other than the held-out component. A candidate passes only above that threshold. Positives and missing-source controls come from the same panel. This is component-disjoint cross-fitting on an exploratory benchmark. It neither certifies a population risk bound nor generalizes to new clue styles.

Common timing includes query encoding, lexical scoring, dense scoring and both graph joins. Gate timings are factored from the observed component timings and charge the extra encoder pass only on expansion candidates. They are not independent service timings. Count exact source-pair recovery and wrong sources before role verification.

## RTX 5090 results

Job 42424 reproduces both original ungated controls. There are 50 both-source cases and 100 missing-source cases. Top-four abstention leaves 15 rank-12 proposals for selective checking, ten exact and five wrong.

| Policy | Exact with both sources / 50 | Wrong with both sources / 50 | Wrong with one source missing / 100 | Extra encoder forwards / 150 | Factored P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linked top-four | 30 | 1 | 3 | 0 | 9.11 ms |
| Linked expansion | 40 | 1 | 8 | 0 | 9.11 ms |
| Dense rank guard | 30 | 1 | 4 | 0 | 9.11 ms |
| Full-clue embedding guard | 31 | 1 | 4 | 0 | 9.11 ms |
| IDF lexical coverage gate | 38 | 1 | 4 | 0 | 9.13 ms |
| Fragment similarity gate | 30 | 1 | 4 | 15 | 17.50 ms |
| Fragment margin gate | 30 | 1 | 4 | 15 | 17.50 ms |

The lexical gate accepts eight exact and one wrong expansion. On result excerpts it recovers three additional exact pairs and introduces no new false expansion. On keywords it accepts all five additional true pairs and one false pair. The earlier selective 0.6B cross-encoder scored 31 exact and four missing-source errors on this same panel with 15 additional cross-encoder calls. That comparison is historical and uses different factored timings.

The extra read of fragment embeddings recovers none of the ten added positives. Lexical coverage is the stronger candidate for a subsequent fixed-threshold replication. Its one residual wrong expansion and the errors already accepted by top-four remain unresolved. The 15 selectively scored proposals are too few for a general claim, and the benchmark has repeatedly supported method development. Freeze a threshold before checking independent sources and clues. [Aggregate results](results.json)

## Reproduce

Use the pinned source-confirmation panel, corpus and verified Qwen embedding checkpoint. The runner checks the original aggregate ranking against the replay. Run inside a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.source_clue_agreement \
    PUBLISHER_PROJECTION SOURCE_PANEL QWEN_EMBEDDING PRIOR_SOURCE_RESULTS NEW_RESULTS
```
