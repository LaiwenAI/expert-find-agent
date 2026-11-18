# External replication of matched ownership-swap supervision

Freeze the four role-mapping arms from the [ownership-swap ablation](../plos-credit-role-swap-v1/README.md), including the synthetic training data, ridge penalty and frozen Qwen3 encoder. Capture a new PLOS ONE article inventory from title searches for speech recognition, sentiment analysis, knowledge graphs, machine reading comprehension and question answering. The snapshot takes up to 60 articles in publication-date order. Filter out any DOI, checksum-valid publisher ORCID, or normalized author name shared with the previous 130-record inventory. The matched-edit protocol retains articles with complete declarations and paired controls.

The [boundary wording](boundary-phrases.json) supplies two newly authored expressions per role and usually places a related contributor's task first. The expressions have no exact normalized overlap with the synthetic training descriptions, NISO examples or either previous transfer wording bank. The previous straightforward transfer wording is repeated as a control. Each group holds the source and authors fixed while switching one role requirement, with a different author as the stable control. The labels refer to publisher-declared roles.

The [manifest](manifest.json) pins the capture, declared contributions and article hashes. The [pairs](pairs.jsonl) store attribution labels and the [source audit](source-audit.json) pins the evaluator's scope. The data inherit the PLOS attribution recorded by DOI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Results

The API returned 60 records. A publisher-license check excluded one. The identity audit excluded 24 shared DOIs, one article with a shared publisher identifier, and one with a shared normalized author name. Nine more lacked usable declarations or matched edits. The retained 24 articles yield 80 groups. The stricter name check was applied before reading outcome data.

Job `40503` evaluates every frozen arm on an RTX 5090. All four arms score 80/80 groups on the straightforward wording. Results on the new boundary wording are:

| Training data | Atomic labels / 28 | Both pairs exact / 80 | Changed pair exact | Stable pair exact | Unsupported promotions | Warm P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original descriptions | 19 | 40 | 50 | 59 | 16 | 8.30 ms |
| Add single-clause controls | 19 | 44 | 50 | 63 | 14 | 8.27 ms |
| Add alternate role swaps | 20 | 45 | 52 | 64 | 10 | 8.31 ms |
| Add confusable role swaps | 20 | 40 | 46 | 59 | 19 | 8.30 ms |

The confusable-role arm fixes six groups and loses six against original descriptions. The paired difference is 0.00 points (descriptive 95% publication-bootstrap interval −7.32 to +7.89). Unsupported promotions rise from 16 to 19. The +8-group improvement on the exploratory cohort fails to replicate with new articles and new wording. Alternate swaps score best on these specific phrases. The timing measures task encoding and role decisions with declarations supplied. Evaluations reuse 28 boundary phrases across the articles, and these phrases have authored labels.

## Reproduce

Use [the builder](../../experiments/credit/build_external_role_edits.py) with its `capture` command for the raw publisher records, then use:

```sh
PYTHONPATH=src:. python -m experiments.credit.build_external_role_edits build \
  NEW_SNAPSHOT PRIOR_130_RECORD_MANIFEST NEW_FIXTURE PRIOR_XML_DIRECTORY [PRIOR_XML_DIRECTORY...]
```

Use the [ownership-swap training replay](../plos-credit-role-swap-v1/README.md#reproduce) with the new fixture, source audit and boundary phrases. Pass `--comparison-phrases benchmarks/plos-credit-transfer-v1/phrases.json` to keep the straightforward wording control fixed. Run the model benchmark in a Slurm CUDA allocation.
