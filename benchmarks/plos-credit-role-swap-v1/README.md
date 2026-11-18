# Matched role-ownership swaps for synthetic supervision

Start from the published [84 task descriptions](../plos-credit-edits-v1/role-training.json). Match potentially confusable roles, including Formal analysis with Software and Conceptualization with Methodology. For each pair of descriptions, hold the task text and order fixed and swap which person is credited. The classification label follows the task credited to the selected contributor. Each matched pair has identical word counts and opposite labels. Two wording templates yield 84 pairs across the 14 roles.

The [generated payload](training.json) includes two count-matched controls. Single-clause augmentation adds the same task text with neutral ownership wording. Alternate role swaps preserve the two-clause format and labels but use a different disjoint role matching. All augmented arms add 168 examples to the original 84. Ridge regularization, frozen Qwen3-Embedding-0.6B, and inference input remain unchanged. Methods use the same base embeddings.

## Results

Slurm job `40498` uses the previously explored [wording stress test](../plos-credit-role-transfer-hard-v1/README.md) with 116 matched groups over 30 articles. Every arm gets 116/116 exact groups on the original transfer wording. The table shows the harder wording:

| Training data | Atomic labels / 28 | Both pairs exact / 116 | Changed pair exact | Stable pair exact | Unsupported promotions | P95 encoding and role decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original descriptions | 25 | 88 | 91 | 103 | 9 | 8.37 ms |
| Add single-clause controls | 24 | 86 | 89 | 104 | 10 | 8.37 ms |
| Add alternate role swaps | 24 | 84 | 88 | 104 | 14 | 8.38 ms |
| Add confusable role swaps | 25 | 96 | 99 | 110 | 8 | 8.40 ms |

The confusable-role arm fixes 11 groups and loses three relative to the original descriptions. The paired difference is +6.90 points, with a descriptive 95% publication-bootstrap interval from 0.00 to +15.25. Compared with alternate role swaps, it fixes 12 groups and loses none. Atomic NISO results are 34, 37, 36, and 36 correct out of 49 in the same table order. The data construction improves this tested compositional workload at nearly unchanged local timing. Wording reappears across papers, and this comparison follows inspection of the same diagnostic cohort. It is an exploratory data ablation.

## Reproduce

Build the matched banks with `python -m experiments.credit.role_swap_training build ROLE_TRAINING_JSON NEW_GENERATED_JSON`. Run [the training replay](../../experiments/credit/role_swap_training.py) on a CUDA compute node within a Slurm allocation:

```sh
PYTHONPATH=src:. python -m experiments.credit.role_swap_training run \
  TRANSFER_FIXTURE SOURCE_AUDIT_JSON BOUNDARY_PHRASES_JSON NISO_EXAMPLES_JSON \
  ROLE_TRAINING_JSON GENERATED_JSON EMBEDDING_CHECKPOINT NEW_RESULT_JSON
```

The audit and boundary phrases come from the [wording stress test](../plos-credit-role-transfer-hard-v1/README.md).
