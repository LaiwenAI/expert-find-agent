# Source Confirmation on Author-Written LitSearch Questions

[LitSearch](https://aclanthology.org/2024.emnlp-main.840/) supplies 597 questions with one or more labeled target papers. This replay indexes every title and abstract in the 64,183-paper clean corpus. It evaluates the 246 questions written by paper authors, with 155 from ACL and 91 from ICLR. The 96 calibration questions are selected by a fixed text hash from the inline-citation subset after excluding shared labeled targets and exact normalized test questions.

Each request runs with all sources present and again with all released targets removed before ranking. The same BM25 statistics are used in both views. An accepted paper outside the benchmark labels is unjudged. Acceptance after withholding measures whether the system substitutes another paper when the labeled originals are unavailable. It does not establish that all substituted papers are irrelevant.

The frozen PLOS gate requires more than eight query terms, a token coverage score of at least 0.75 and a margin of 0.30 among four BM25 candidates. The new controls calibrate token coverage or Qwen3-Reranker-0.6B on the inline questions. The reranker scores six BM25 candidates from title and abstract. Policy selection maximizes accepted known targets subject to budgets for unjudged selections in each calibration view separately. The two budgets are 5% and 10% of calibration requests. The test set is evaluated at the resulting fixed thresholds.

| Route | Present known target / 246 | Present unjudged selection | Withheld acceptance / 246 | Present P95 |
| --- | ---: | ---: | ---: | ---: |
| BM25 top result | 99 | 147 | 246 | 13.37 ms |
| Rerank and always select | 113 | 133 | 246 | 70.41 ms |
| Frozen PLOS coverage | 5 | 1 | 1 | 13.67 ms |
| Coverage, 5% calibration budget | 0 | 0 | 0 | 13.67 ms |
| Coverage, 10% calibration budget | 45 | 59 | 89 | 13.67 ms |
| Reranker, 5% calibration budget | 12 | 2 | 3 | 70.41 ms |
| Reranker, 10% calibration budget | 42 | 4 | 10 | 70.41 ms |

BM25 top six exposes at least one known target on 148/246 questions. The coverage 5% setting finds no nonempty feasible policy on calibration. The 10% setting accepts 89 substitutions on test, above its calibration fraction of 9/96. The reranker at the 10% setting accepts 10 substitutions on test while retaining 42 known targets. These are observations against released labels. They do not measure true open-world error or replace independently judged negative queries.

The PLOS lexical threshold transfers poorly to author questions. The results separate missing candidate exposure from source confirmation after retrieval. Job `40422` runs a title-and-abstract BM25 index and the frozen reranker on RTX 5090. Timing covers the warm local path, with P95 for every request, including abstentions. The index build takes 17.03 seconds and is reported separately. The comparison leaves corpus-wide dense retrieval and full-paper confirmation for additional testing.

## Full-corpus dense and fused candidates

Use the same documents, calibration rows and test questions. The frozen Qwen3-Embedding-0.6B model encodes every title and abstract once for the index. Dense search scores the full resident matrix in FP32 and includes live query encoding in the timer. Reciprocal rank fusion combines the first 24 hits per channel at constant 60. All three retrieval routes pass six candidates to the same reranker. The run reports unfiltered ranking, a shared reranker threshold from job `40422`, and route-specific thresholds fitted on the identical calibration split. No cross-request cache reuse is measured.

| Candidates | Known target in six / 246 | Initial top one | Reranked top one | Retrieval P95 | Including reranking P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 148 | 99 | 113 | 13.70 ms | 70.63 ms |
| Exact dense | 177 | 106 | 131 | 7.34 ms | 65.05 ms |
| Fusion | 175 | 119 | 129 | 21.18 ms | 78.30 ms |

| Candidates | Shared threshold, known / withheld accepts | Route-specific threshold, known / withheld accepts |
| --- | ---: | ---: |
| BM25 | 42 / 10 | 42 / 10 |
| Exact dense | 43 / 10 | 28 / 7 |
| Fusion | 43 / 10 | 49 / 12 |

Dense retrieval increases coverage by 11.79 percentage points relative to BM25 (target-connected bootstrap 95% interval +5.31 to +17.89). The shared verification threshold returns only one additional known target. Recalibration can further trade known-target recovery against accepting unjudged alternatives. The corpus and labels are the same in both tables. The integrated comparison uses the original BM25 selection path. The array comparison below measures the cheaper equivalent component separately.

Jobs `40426` and `40428` use an RTX 5090. Building the dense vectors takes 196.21 seconds and uses 262,893,568 bytes for the resident matrix. Cold preparation, transport, full-paper evidence and author attribution are outside the measured path.

Build [the dense cache](../../experiments/litsearch/dense_cache.py) with `PROJECTION EMBEDDING_CHECKPOINT NEW_VECTORS.npz`. Run [the candidate comparison](../../experiments/litsearch/dense_retrieval.py) with `PROJECTION VECTORS EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON`. The reader verifies source-ID ordering and the frozen checkpoint and projection hashes.

## Exact BM25 selection

The [BM25 selection replay](../../experiments/litsearch/bm25_selection.py) compares the original Python score dictionary and heap against partitioning the resident arrays. It retains the same postings, scores and numeric document-ID tie break. Both target-present and target-withheld views are tested at depths six and 24. Run with `PROJECTION NEW_RESULT_JSON`.

| Selection depth | Heap P50 / P95 | Array P50 / P95 | Exact ordered matches |
| --- | ---: | ---: | ---: |
| 6 | 12.69 / 13.28 ms | 0.94 / 1.44 ms | 492 / 492 |
| 24 | 12.69 / 13.30 ms | 0.94 / 1.44 ms | 492 / 492 |

Job `40431` alternates both implementations on the same resident 64,183-paper index. This is a retrieval-component timing, including scoring, selection, and target masking. It makes no change to relevance or evidence and excludes request verification.

## Learned source gate and two-paper confirmation

Keep exact dense top six fixed. Fit a seven-feature linear head over dense similarity and rank, resident BM25 scores, and title and abstract token overlap. The 128 training questions come from the inline subset and exclude all labeled targets and exact normalized questions from both calibration and test. Labels encode known targets versus unjudged candidates, including distractors exposed after target withholding. The embedding and reranker checkpoints stay frozen.

Compare the dense cosine gate, the learned gate, reranking the learned head's top two, always reranking all six, and a live cascade. Each standalone method uses the same 10% calibration budgets for the two views. The cascade fits a cheap acceptance gate at 5%, then selects a two-paper backup threshold to keep the combined calibration counts within 10%. Test requests execute the chosen path live. Each source-withheld acceptance still counts as an unjudged alternative.

| Policy | Known top one before gating | Known returned | Unjudged returned | Returned after withholding | Reranker calls | Present / withheld P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cosine gate | 106 | 64 | 21 | 40 | 0 / 0 | 7.30 / 7.33 ms |
| Learned gate | 117 | 55 | 11 | 27 | 0 / 0 | 8.69 / 8.74 ms |
| Rerank two | 128 | 47 | 9 | 17 | 246 / 246 | 26.08 / 25.65 ms |
| Rerank six | 131 | 28 | 5 | 7 | 246 / 246 | 65.01 / 72.17 ms |
| Learned gate, then rerank two | 126 | 67 | 16 | 32 | 191 / 226 | 23.78 / 25.14 ms |

Each view has 246 author-written requests. All arms retrieve identical ordered pools and cover 177 known targets with the targets present. The rerank-two and live cascade routes reduce present-view P95 by 60% and 63%, respectively, relative to rerank six. Rerank two returns 24 new known targets and loses five returned by rerank six. It also returns 17 unjudged alternatives after withholding, compared with seven for rerank six. The cascade returns 32 after withholding, exceeding its 10% calibration ceiling on the author-written test. These policies have different test operating points. Equal-risk speedup requires an independent quality check at matched acceptance, including adjudication of alternatives. [Cascade ranking](https://doi.org/10.1145/2009916.2009934) supplies the efficiency precedent.

Job `40441` uses an RTX 5090. Head optimization takes 0.20 s over 1,536 candidate-view examples (84 labeled target occurrences), excluding feature materialization. Train, calibration, and test have mutually disjoint labeled target IDs. Request timing includes live query encoding, search, and reranker calls. It excludes initialization, transport, full-text retrieval, and personal attribution.

### Dense top-two control and threshold sensitivity

Job `40446` adds a fixed dense top-two control and refits thresholds on the same 96 calibration questions at four ceilings. The original arms reproduce every test decision score and candidate order. Each table cell shows **known returned / unjudged returned with targets present / returned after withholding** on the 246 author-written questions.

| Policy | 2% ceiling | 5% ceiling | 10% ceiling | 15% ceiling | Present P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense cosine gate | 32 / 4 / 6 | 55 / 13 / 30 | 64 / 21 / 40 | 74 / 24 / 52 | 7.42 ms |
| Learned gate | 10 / 1 / 3 | 45 / 10 / 20 | 55 / 11 / 27 | 64 / 14 / 35 | 8.83 ms |
| Rerank dense top two | 23 / 2 / 3 | 30 / 7 / 9 | 42 / 8 / 14 | 47 / 13 / 20 | 25.02 ms |
| Rerank learned top two | 18 / 2 / 4 | 13 / 3 / 4 | 47 / 9 / 17 | 55 / 12 / 22 | 26.32 ms |
| Rerank top six | 21 / 2 / 4 | 32 / 3 / 5 | 28 / 5 / 7 | 32 / 10 / 13 | 65.01 ms |

At the 2% calibration setting, dense top two uses 62% less P95 latency and returns two more known targets than top six with one fewer withheld-view return. The paired difference in known-target return rate is +0.81 points (descriptive 95% interval −2.03 to +3.66, bootstrap over 245 target-connected groups). The ordering changes at other thresholds. At 5%, learned top two returns 19 fewer known targets than top six at similar withheld counts. Dense top two is therefore a necessary control for the learned selector. The cohort was previously explored, and the withheld-view counts measure incomplete-label behavior. A noninferiority claim needs fresh adjudication and a margin fixed before testing.

Run [the learned replay](../../experiments/litsearch/learned_confirmation.py) with `PROJECTION VECTORS EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON`.

## Reproduce

Download [the pinned query Parquet](https://huggingface.co/datasets/princeton-nlp/LitSearch/resolve/9573fb284a1026c998df47024b888a163f0f0e25/query/full-00000-of-00001.parquet). Run [the projection tool](../../experiments/litsearch/prepare_titles.py) with `QUERY_PARQUET NEW_PROJECTION.json.gz` in an environment with PyArrow. It reads only the three selected columns from the six corpus shards using range requests. Full article bodies and third-party dataset rows are omitted from the release.

Run [the GPU replay](../../experiments/litsearch/source_withholding.py) with `PROJECTION QWEN3_RERANKER_CHECKPOINT NEW_RESULT_JSON` in the research environment. The replay excludes full article bodies and performs no author identity or contributor-role judgment.
