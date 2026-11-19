# Full-history priors on the matched SE-PEF text panel

Use the exact 256 validation and 512 test questions selected for [the text benchmark](../sepef-temporal-intake-v1/README.md). Load the official pre-2020 expert snapshot. Rank by full answer count, globally or within the known Stack Exchange community. Retain the latest owned historical answer as the citation for the prior. An owned answer establishes provenance. Its relevance to the query remains unjudged.

| Prior | Validation Hit@5 | Test Hit@5 | Test MRR@10 |
| --- | ---: | ---: | ---: |
| Global full history | 6.25% | 7.62% | 0.0465 |
| Community full history | 34.38% | 38.09% | 0.2413 |

These scores are the label-matched text-free controls. The preparation program pins the query files and checks the historical profile boundary. It writes the prior index locally and publishes only aggregate metrics.

## Frozen text fusion procedure

Score the same capped answer texts using the frozen dense community route. Combine its first 50 experts with the first 50 from the full-history prior using weighted reciprocal-rank fusion with offset 60. Select the prior weight from 0, 0.25, 0.5, 1, 2 and 4 on validation MRR@10. Break ties using Hit@5 and then the smaller weight. Measure prior, dense and fused first-query paths separately on test. Report paired changes by whole-community bootstrap. Query cleanup, online encoding when applicable, ranking and serialization count towards request time.

Validation selected the largest tested weight, 4. The grid tests finite mixtures. The prior-only arm is evaluated directly on test.

| Test route | Hit@5 | MRR@10 | P95 (ms) |
| --- | ---: | ---: | ---: |
| Full-history community prior | 38.09% | 0.2413 | 0.010 |
| Dense community | 8.59% | 0.0513 | 14.11 |
| Weighted fusion | 35.55% | 0.2079 | 14.25 |

Relative to the prior, fusion changed Hit@5 by -2.54 percentage points (95% community-bootstrap interval -7.20 to +2.94) and MRR@10 by -0.0333 (interval -0.0708 to +0.0038). This experiment did not establish a quality gain over the cheaper control. The timing is warm, sequential, in-process request execution. The label identifies the released answerer. The evaluation checks rank and historical answer ownership, with no human judgment of the selected source's relevance.

[Aggregate fusion results](fusion-results.json)

## Lexical agreement gate

A proposed router uses the overlap of the lexical and historical-prior top five to choose cheap lexical fusion or dense fusion. Trying the first nonzero threshold would encode 73.05% of validation queries. The 50% validation budget therefore selected no dense calls. The query-independent control also used zero dense calls at this budget.

| Test route | Hit@5 | MRR@10 | P95 (ms) | Encoded queries |
| --- | ---: | ---: | ---: | ---: |
| Community prior | 38.09% | 0.2413 | 0.015 | 0% |
| Always dense fusion | 35.55% | 0.2079 | 26.16 | 100% |
| Agreement gate, reduced to lexical fusion | 24.80% | 0.1702 | 16.20 | 0% |

In this run the gate did not provide a usable budgeted action: taking any additional dense queries by this rule would exceed its validation cap. Relative to the prior, the resulting lexical fusion lost 13.28 percentage points of Hit@5 (95% community-bootstrap interval -20.76 to -4.85). Comparisons involving query time are within each run. [Aggregate router results](router-results.json)

## Reproduce

Use the `expert_test_data.json` member and the panel from the linked intake. Run preparation on a Slurm compute node and scoring in a Slurm GPU allocation:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.sepef_prior_fusion prepare \
  EXPERT_TEST_DATA PANEL_DIR LOCAL_PRIOR_JSON NEW_METADATA_RESULTS
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.sepef_prior_fusion run \
  PANEL_DIR LOCAL_PRIOR_JSON QWEN_EMBEDDING_CHECKPOINT NEW_GPU_RESULTS
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.sepef_query_router \
  PANEL_DIR LOCAL_PRIOR_JSON NEW_GPU_RESULTS QWEN_EMBEDDING_CHECKPOINT \
  SCRATCH_DIR NEW_ROUTER_RESULTS
```

[Aggregate results](results.json) include query, history and program hashes. The release excludes the prior index, IDs and source text.
