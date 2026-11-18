# CSFCube transfer of source confirmation

[CSFCube](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/20f07591c6fcb220ffe637cda29bb3f6-Abstract-round2.html) provides expert-adjudicated relevance grades for scientific query-by-example. This experiment checks whether the source selector and acceptance thresholds fitted on LitSearch transfer to facet-specific queries. It uses the 50 query-facet pairs and their released candidate pools, excludes the query paper itself, and presents title and abstract text to the frozen encoder and reranker. No CSFCube labels enter training or calibration.

Define strong relevance as adjudicated grade 2 or 3. The present view serves the original judged pool, and the withheld view removes every strongly relevant candidate. Grade 0 and grade 1 returns are reported separately to distinguish judged irrelevance from weak facet similarity. Search ranks the remaining judged candidates by dense cosine similarity and passes the top six to each policy. The measured behavior covers ranking and acceptance inside preselected pools.

## Results

Job `40451` runs on an RTX 5090. A strong candidate appears in the shared top six for 45 of 50 queries. The freeze uses score and gap thresholds fitted at the 10% LitSearch calibration ceiling. Results are counts out of 50 per view.

| Policy | Strong top one before gating | Returned grades 2–3 | Returned grade 1 / 0 | Withheld returns grade 1 / 0 |
| --- | ---: | ---: | ---: | ---: |
| Dense cosine gate | 28 | 11 | 4 / 1 | 6 / 4 |
| Learned gate | 28 | 14 | 2 / 3 | 9 / 4 |
| Rerank dense top two | 29 | 2 | 0 / 0 | 0 / 0 |
| Rerank learned top two | 28 | 0 | 2 / 0 | 1 / 1 |
| Rerank top six | 28 | 2 | 0 / 0 | 0 / 0 |
| Learned gate, then rerank two | 30 | 14 | 2 / 3 | 8 / 4 |

At the stricter 2% LitSearch ceiling, all three standalone reranker gates abstain on all 50 requests in both views. Dense and learned gates return 5 and 14 strong candidates, and return 1 and 2 grade-zero candidates in the present view, respectively. The learned top-two selector shows no ranking gain over dense top two on this pool. Acceptance drops sharply for the reranker even where the first result is judged strongly relevant. This separates score-threshold transport from ranking quality. It supports using an in-domain calibration set when evaluating a deployment route. These results are specific to faceted paper-level relevance and preset candidate pools.

## Grouped in-domain calibration

The [calibration replay](../../experiments/csfcube/grouped_calibration.py) keeps all facets of a query paper together. For each of five folds, it fits score and gap thresholds using the other four folds, with separate ceilings on grade 0–1 returns in the present view and any return in the withheld view. The head, encoder, reranker, and candidate pools remain frozen. At the 10% calibration ceiling, out-of-fold counts are:

| Policy | Strong returns | Grade 1 / 0 returns | Withheld grade 1 / 0 returns | Present P95 |
| --- | ---: | ---: | ---: | ---: |
| Dense cosine gate | 10 | 1 / 1 | 1 / 2 | 7.61 ms |
| Learned gate | 12 | 1 / 1 | 2 / 2 | 8.41 ms |
| Rerank dense top two | 13 | 3 / 1 | 4 / 2 | 25.79 ms |
| Rerank learned top two | 14 | 3 / 1 | 4 / 2 | 26.51 ms |
| Rerank top six | 17 | 2 / 1 | 2 / 3 | 53.93 ms |
| Rerank dense top two, require agreement with cosine | 12 | 2 / 1 | 3 / 2 | 25.79 ms |

At 10%, dense top two recovers 13 strong returns after local calibration, compared with 2 under the transported threshold. It remains four returns below top six, a paired difference of −8.0 percentage points (descriptive 95% query-paper bootstrap interval −19.6 to +2.0). The learned shortlist adds one strong return relative to dense top two, and the agreement gate loses one. Dense top two also returns one more result with all strong candidates withheld. This replay establishes neither quality noninferiority nor a deployment error bound. At the 5% and 20% calibration ceilings, dense top two returns 4 and 16 strong candidates, compared with 6 and 20 for top six.

Run the calibration replay with `DATA_DIR FROZEN_CSFCUBE_RESULT_JSON NEW_RESULT_JSON`. P95 comes from the original measured GPU requests. The agreement check reuses the two-paper observation and omits comparison overhead.

## Adaptive depth on judged candidate pools

Job `40574` selects depth before one cross-encoder call. The experiment preserves the earlier five query-paper folds and stages. For each fold, fit the dense-gap split and two acceptance rules on the other four folds. Fixed routes use the same score and margin grid as the adaptive stages. The hash route chooses depth without retrieval scores. Every held-out route runs live, with source-present and strong-source-withheld views. Dense, top-two, and top-six pool membership and decision scores match the previous measurement in all 100 views.

Cells show **strong returns with sources present / grade-zero returns after withholding / grade-one returns after withholding** across 50 queries. P95 is for the source-present 10% setting.

| Method | 5% ceiling | 10% ceiling | 20% ceiling | Present P95 |
| --- | ---: | ---: | ---: | ---: |
| Dense score and gap | 9 / 1 / 0 | 10 / 2 / 0 | 15 / 3 / 7 | 7.68 ms |
| Verify dense top hit | 8 / 2 / 2 | 9 / 2 / 6 | 16 / 3 / 11 | 16.54 ms |
| Rerank top two | 1 / 1 / 3 | 12 / 2 / 3 | 16 / 2 / 6 | 25.88 ms |
| Rerank top six | 5 / 2 / 2 | 15 / 2 / 3 | 19 / 4 / 7 | 53.90 ms |
| Dense-gap routing | 12 / 2 / 2 | 15 / 2 / 4 | 15 / 3 / 7 | 23.23 ms |
| Query-hash routing | 7 / 2 / 3 | 15 / 2 / 8 | 13 / 3 / 6 | 25.77 ms |

At 10%, the gap router and top six each return 15 strong papers. The gap router's P95 is 57% lower, and it returns one additional weakly relevant paper in the withheld view. It also returns five grade 0–1 papers in the present view, compared with two for top six. At 20%, the gap router returns four fewer strong papers. This controlled comparison distinguishes weak matches and judged irrelevance. The small out-of-fold study and its preselected pools leave open-world equal-risk deployment unverified. [Results and grouped intervals](adaptive-depth-v1.json)

Run [the live replay](../../experiments/csfcube/adaptive_depth.py) with `DATA_DIR RAW_FROZEN_CSFCUBE_RESULT_JSON EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON` on a Slurm compute node.

## Reproduce

Fetch the four source files from [CSFCube revision `7ffe012`](https://github.com/iesl/CSFCube/tree/7ffe012c1d92c470f185fc086086caf5b8e05ef3): `abstracts-csfcube-preds.jsonl` and the three `test-pid2anns-csfcube-*.json` files. Run [the transfer test](../../experiments/csfcube/frozen_confirmation.py) with `DATA_DIR LITSEARCH_DEPTH_RESULT_JSON EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON`. The LitSearch input comes from the [depth-controlled replay](../litsearch-source-confirmation/README.md#dense-top-two-control-and-threshold-sensitivity). Dataset files remain under the upstream license and are retrieved separately.
