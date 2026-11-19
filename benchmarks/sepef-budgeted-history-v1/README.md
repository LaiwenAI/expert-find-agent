# Community reservation under a fixed SE-PEF history budget

Use the same published pre-2020 history snapshot and all released validation and test qrels as the [SE-PEF intake](../sepef-temporal-intake-v1/README.md). For each expert, keep an equal number of answers under three policies. Recency takes the latest answers. Community reservation first takes one latest answer per retained community and fills unused slots by recency. When an expert has more communities than slots, prioritize either the most recently active communities or the ones with the most historical answers.

For each question, restrict candidates to experts with at least one retained answer in its community. Rank them by the full-history community answer counts, identical for every policy. This separates candidate availability from the loss of prior information caused by counting only retained answers. A label is reachable when at least one labeled expert has history in the restricted scope. Availability describes answer ownership in the archive. It does not measure whether the selected answer can justify a recommendation.

## Test results

| Answers per expert | Total answers | Recency label reachable | Reserve recent communities | Reserve active communities |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 13,081 | 16,502 | 17,506 | 17,631 |
| 8 | 26,123 | 17,020 | 17,998 | 18,033 |
| 16 | 51,889 | 17,371 | 18,092 | 18,095 |

The test split has 19,995 questions. The full-history community scope reaches 18,100. At eight slots, reserving the most active communities recovers 1,013 reachable questions versus recency. Applying the same historical prior changes Hit@5 from 38.83% to 38.79% and MRR@10 from 0.2446 to 0.2455. The added availability barely changes this prior's top-ranked predictions. These results support using community reservation as an equal-budget text retrieval control. The [matched text panel](../sepef-source-coverage-v1/README.md) tests whether the added coverage improves ranking.

## Reproduce

On a Slurm compute node with the three pinned metadata members from the intake:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.sepef_budgeted_history \
  METADATA_DIR NEW_RESULT_JSON
```

[Aggregate results](results.json) include the validation split, source hashes and the exact equal-budget counts. Raw queries, authors and answers stay outside the source release.
