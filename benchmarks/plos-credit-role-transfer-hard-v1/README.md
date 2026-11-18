# Role decision transfer under confusing wording

Select 30 articles and 116 matched requirement-edit groups from the [PLOS transfer fixture](../plos-credit-transfer-v1/README.md). The [source audit](source-audit.json) excludes three shared DOIs and an additional article linked by a publisher ORCID to the original edit or compositional fixtures. Authors without published identifiers can still recur.

The original transfer wording and the [28 newly authored phrases](phrases.json) render the same role requirements for the same authors. The new phrases mention related roles in exclusions and incidental context. For example, a phrase may attribute statistical analysis while assigning software implementation to someone else. Each phrase has a single predetermined intended role. Normalized strings do not overlap the 84 training descriptions, the 49 NISO examples, or the other wording bank. The new wording is a controlled stress test with authored labels.

Restore the original-seed trained heads and adapters from [the compositional training replay](../plos-credit-compositional-edits-v1/README.md#small-model-training), without refitting them or selecting a seed on this fixture. Compare with both frozen ridge controls.

## Results

Job `40492` used an RTX 5090. All four methods classify all 28 original transfer phrases correctly and match both authors' labels before and after the edit in all 116 original-wording groups. The same groups rendered with the new boundary wording show:

| Route | Atomic roles / 28 | Both pairs exact / 116 | Changed pair exact / 116 | Stable pair exact / 116 | Unsupported promotions | Warm P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen ridge, 70 descriptions | 22 | 82 | 86 | 104 | 14 | 8.34 ms |
| Frozen ridge, 84 descriptions | 25 | 88 | 91 | 103 | 9 | 8.28 ms |
| Trained head | 25 | 87 | 90 | 103 | 9 | 8.35 ms |
| LoRA and trained head | 23 | 78 | 81 | 103 | 13 | 8.51 ms |

LoRA drops from 116 to 78 exact groups across wording, and retrieves ten fewer exact groups than the ridge trained on all 84 descriptions under the boundary wording. The paired difference is −8.62 percentage points with a descriptive 95% publication-bootstrap interval of −15.25 to −2.54 points. The wording bank is small and repeated across papers. A paper bootstrap does not capture uncertainty over new phrasing. Against the matched 70-description ridge, LoRA loses four exact groups and produces one fewer unsupported promotion. Timing covers encoding two requirement versions as one batch and selecting their role labels, with source declarations already supplied.

## Reproduce

Rebuild the source audit with `python -m experiments.credit.replay_role_training_transfer audit ORIGINAL_EDIT_FIXTURE TRANSFER_FIXTURE COMPOSITION_FIXTURE NEW_AUDIT_JSON XML_DIRECTORY [XML_DIRECTORY...]`. The input XML inventory comes from the [cross-source fixture](../plos-credit-cross-source-v1/README.md).

Run [the frozen transfer replay](../../experiments/credit/replay_role_training_transfer.py) with:

```sh
PYTHONPATH=src:. python -m experiments.credit.replay_role_training_transfer run \
  TRANSFER_FIXTURE COMPOSITION_FIXTURE SOURCE_AUDIT_JSON BOUNDARY_PHRASES_JSON \
  NISO_EXAMPLES_JSON ROLE_TRAINING_JSON EMBEDDING_CHECKPOINT ORIGINAL_SEED_OUTPUT NEW_RESULT_JSON
```

The original-seed output contains the two verified adapter/head files and their training results. Run model inference inside a CUDA Slurm allocation.
