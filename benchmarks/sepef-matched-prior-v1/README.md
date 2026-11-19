# Full-history priors on the matched SE-PEF text panel

Use the exact 256 validation and 512 test questions selected for [the text benchmark](../sepef-temporal-intake-v1/README.md). Load the official pre-2020 expert snapshot. Rank by full answer count, globally or within the known Stack Exchange community. Retain the latest owned historical answer as the citation for the prior. An owned answer establishes provenance. Its relevance to the query remains unjudged.

| Prior | Validation Hit@5 | Test Hit@5 | Test MRR@10 |
| --- | ---: | ---: | ---: |
| Global full history | 6.25% | 7.62% | 0.0465 |
| Community full history | 34.38% | 38.09% | 0.2413 |

These scores are the label-matched text-free controls. The preparation program pins the query files and checks the historical profile boundary. It writes the prior index locally and publishes only aggregate metrics.

## Frozen text fusion procedure

Score the same capped answer texts using the frozen dense community route. Combine its first 50 experts with the first 50 from the full-history prior using weighted reciprocal-rank fusion with offset 60. Select the prior weight from 0, 0.25, 0.5, 1, 2 and 4 on validation MRR@10. Break ties using Hit@5 and then the smaller weight. Measure prior, dense and fused first-query paths separately on test. Report paired changes by whole-community bootstrap. Query cleanup, online encoding when applicable, ranking and serialization count towards request time.

The fusion implementation has passed a synthetic GPU smoke test. Official text-fusion measurements are pending the answer archive extraction and the first-query GPU run.

## Reproduce

Use the `expert_test_data.json` member and the panel from the linked intake. Run preparation on a Slurm compute node and scoring in a Slurm GPU allocation:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.sepef_prior_fusion prepare \
  EXPERT_TEST_DATA PANEL_DIR LOCAL_PRIOR_JSON NEW_METADATA_RESULTS
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.sepef_prior_fusion run \
  PANEL_DIR LOCAL_PRIOR_JSON QWEN_EMBEDDING_CHECKPOINT NEW_GPU_RESULTS
```

[Aggregate results](results.json) include query, history and program hashes. The release excludes the prior index, IDs and source text.
