# Author-scoped lightweight verification across publications

Use the existing [PLOS cross-publication contributor edits](../plos-credit-compositional-edits-v1/README.md). Read publisher-declared roles and prelinked author identifiers. Keep the published split and NISO phrase selection. Test labels come from publisher declarations with controlled single-role edits, missing records, or known different authors. These are specified contribution roles in named sources.

Compare one joint check over paper-pooled roles, one joint check over the target author's roles, two or fewer author-scoped checks grouped by source, and up to three author-scoped checks grouped by task. Use the same identity and missing-source rules in all four. Reject a confirmed distinct person without a model call. Missing sources block recommendation and still allow rejection from the visible source. Use the minimum entailment margin across visible checks. Margins strictly below the lower threshold reject, while a complete same-person record strictly above the upper threshold supports. Otherwise abstain.

Compare the frozen MiniLM and both [SciFact rationale-pair checkpoints](../scifact-minilm-window-supervision-v1/README.md). Fit thresholds on the 108 calibration views. The upper threshold permits at most 5% false promotions, under two separately evaluated denominators. Global calibration includes all 84 non-support decisions. Eligible-only calibration uses the 12 edited requests whose identity and source checks allow promotion. The lower threshold allows at most 5% wrong rejection among valid supports and unresolved cases. Keep thresholds fixed on 156 held-out views. The eligible calibration group has only 12 negative edits and therefore permits zero calibration errors. Test views share three publication components.

## RTX 5090 results

The verified run (Slurm 42323) reproduces the global decisions from job 42315 and adds the eligible-only gate. In the table, true support is counted among 30 complete positives. Wrong-role support is counted among the 18 complete same-person edits whose focal role belongs to a coauthor. Every method structurally blocks the 36 known-different-person cases and the missing-source requests from promotion. Ranges span the two SciFact seeds.

| Reader and partition | True supports, global / eligible-only | Wrong-role supports, global / eligible-only | Component P95 |
| --- | ---: | ---: | ---: |
| Frozen, paper-pooled joint | 3 / 0 | 1 / 0 | 1.32 ms |
| Frozen, author-scoped joint | 12 / 2 | 9 / 3 | 1.25 ms |
| Frozen, author-scoped by source | 21 / 0 | 13 / 0 | 2.32 ms |
| Frozen, author-scoped by task | 21 / 0 | 12 / 0 | 3.46 ms |
| SciFact, author-scoped joint | 7–8 / 3 | 5–8 / 3 | 1.23–1.25 ms |
| SciFact, author-scoped by source | 24 / 0 | 15 / 0 | 2.30–2.32 ms |
| SciFact, author-scoped by task | 24 / 1 | 12 / 0 | 3.39–3.41 ms |

At the global gate, source partitioning is faster than task partitioning and retains the same number of supports. It incorrectly accepts 13–15 of the 18 complete wrong-role edits. The stricter gate reduces that error only at the cost of accepting none of the true supports in the source-partitioned setting. Both SciFact checkpoints still accept three wrong-role edits under joint checking, despite the zero-error calibration gate. Joint checking and source partitioning correctly handle at most one of the 18 present edit pairs. Among prior methods on the same fixture, the factored taxonomy head plus structured checks handled 7/18 of those pairs and retained 18/30 true supports with five wrong promotions across all views. Its costs were measured in a separate run.

These decisions concern published role declarations and fixed source locations. The small calibration cohort cannot certify a 5% population risk bound. Component timing sums pre-rendered views' tokenization and sequential inference. It excludes query decomposition, source retrieval, rendering and transport. Preserve the structured factored comparator. Any semantic fallback should be tested specifically on candidates eligible for recommendation, with separate reporting of wrong-role acceptance, abstention and good candidates lost. [Aggregate results](results.json)

## Reproduce

Get the pinned NISO examples and model checkpoints as in the [source benchmark](../plos-credit-compositional-edits-v1/README.md#reproduce) and the [MiniLM benchmark](../scifact-minilm-window-supervision-v1/README.md). Run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.partitioned_nli_verification \
    FIXTURE NISO_EXAMPLES_JSON SCIFACT_REPORT SCIFACT_WEIGHTS MINILM_CHECKPOINT NEW_RESULTS
```
