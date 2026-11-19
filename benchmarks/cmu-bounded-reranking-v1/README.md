# Bounded cross-encoder evidence search on CMU Gold

This study uses the `d_20_1` reviewer histories from the [CMU Gold release at revision 0029673](https://github.com/niharshah/goldstandard-reviewer-paper-match/tree/00296730e3ebe058c7b4cc567cd52208ca80735c). The [dataset paper](https://openreview.net/pdf/42c1da840018dbd09102b5e4c958e7c38d761ab0.pdf) uses reviewers' judgments of their expertise to evaluate relative paper suitability within each reviewer. We use all labeled cases and exclude owned publications if their ID or normalized title matches the query paper. Each route receives the same reviewer and history.

For each source, let `d` be embedding cosine and `m` be the reranker's yes-minus-no logit margin. The combined score is `d + 0.10 sigmoid(m)`. The reviewer score is the maximum source score. The upper bound for an unchecked source is `d + 0.10`; the highest dense score is a lower bound before making any cross-encoder call. The bonus is fixed for this pilot.

| Route | Evidence reads |
| --- | --- |
| Dense | Maximum owned-publication cosine |
| First two / first five | One batch of the highest-cosine two or five sources, then combined maximum |
| Cross only | Maximum reranker margin after checking the entire history in batches of up to eight |
| Full combined | Check the entire history with the same batch limit, then take the combined maximum |
| Two-source start | Check the first two, then stop when the remaining upper bounds fall below the best score seen |
| Upfront bound | Discard sources whose upper bound is below the initial dense lower bound, check up to eight in the first batch, and update the bound between remaining batches |

Bounded methods process the same dense order and preserve its tie order. With fixed per-source scores, the maximum is unchanged by skipping a source whose upper bound is below an already attainable value. Changing GPU batch shape can change the underlying logits, so the benchmark also compares winning source IDs, numerical score differences and the ordering of every annotated unequal-grade paper pair.

Both checkpoints are open-weight Qwen3 0.6B models. Histories are embedded offline. The warm per-pair timer includes live query tokenization, query embedding, cosine, source selection, cross-encoder prompt assembly and forward passes. It excludes service transport and online index loading. Timing arms run independently with rotated order.

## Results on RTX 5090

Slurm job 41579 evaluated all 477 labeled reviewer–paper pairs from 58 reviewers and 463 distinct papers. Pairwise loss uses the 1,653 within-reviewer pairs with unequal grades, weighted by the grade difference. Lower is better. The resident index embedded 799 publications in 3.61 s.

| Route | Weighted pairwise loss | Mean checks / calls | P95 | Same winning source as full | Changed comparable orders vs full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 23.45% | 0.00 / 0.00 | 7.80 ms | 457/477 | 55 |
| First two | 23.68% | 2.00 / 1.00 | 28.55 ms | 466/477 | 7 |
| First five | 23.54% | 4.85 / 1.00 | 59.10 ms | 476/477 | 2 |
| Cross only | 28.06% | 14.61 / 2.26 | 195.84 ms | 261/477 | 310 |
| Full combined | 23.57% | 14.61 / 2.26 | 195.19 ms | 477/477 | 0 |
| Two-source start | 23.55% | 4.65 / 1.66 | 92.76 ms | 474/477 | 3 |
| Upfront bound | 23.58% | 4.80 / 1.14 | 94.40 ms | 477/477 | 3 |

The two-source start saves 68.2% of checks and 52.5% of P95 relative to full combined scoring. Fixed depth five is faster and changes fewer labeled orders relative to full scoring. Dense produces the lowest observed human-label loss and the lowest latency. Upfront pruning replicates all full-route winning source IDs despite three order changes, so matching evidence IDs alone would overstate output agreement. Model inputs share a 1,024-token cap, which affects 83 scored rows in each exhaustive arm. This experiment measures within-reviewer ordering on one history realization. It has no paper-level citation-support labels, no global candidate retrieval, and no service-load measurement. [Aggregates](results.json)

## Reproduce

Obtain the upstream data locally under its [data terms](https://github.com/niharshah/goldstandard-reviewer-paper-match/blob/00296730e3ebe058c7b4cc567cd52208ca80735c/LICENSE-DATA.md). The release contains only aggregate measurements, without paper metadata, abstracts or reviewer IDs. Use a Slurm CUDA allocation and pinned checkpoints with verified provenance files.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.decision.cmu_bounded_reranking \
  CMU_GOLD_ROOT EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```
