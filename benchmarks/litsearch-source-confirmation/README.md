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

## Reproduce

Download [the pinned query Parquet](https://huggingface.co/datasets/princeton-nlp/LitSearch/resolve/9573fb284a1026c998df47024b888a163f0f0e25/query/full-00000-of-00001.parquet). Run [the projection tool](../../experiments/litsearch/prepare_titles.py) with `QUERY_PARQUET NEW_PROJECTION.json.gz` in an environment with PyArrow. It reads only the three selected columns from the six corpus shards using range requests. Full article bodies and third-party dataset rows are omitted from the release.

Run [the GPU replay](../../experiments/litsearch/source_withholding.py) with `PROJECTION QWEN3_RERANKER_CHECKPOINT NEW_RESULT_JSON` in the research environment. The replay excludes full article bodies and performs no author identity or contributor-role judgment.
