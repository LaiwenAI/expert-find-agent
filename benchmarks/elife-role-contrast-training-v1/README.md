# Compare synthetic role contrasts on frozen source pairs

The [option-calibration experiment](../elife-role-option-calibration-v1)
found a role recognition gap even after correcting option priors. Here
the source pair, byline links, attribution checks, role wording, and
frozen Qwen models remain the same. The configurable component is the
role classifier. A linear head reuses the frozen Qwen 0.6B activity
vector, without adding an embedding or 4B call.

Start with the [84 existing role descriptions](../plos-credit-edits-v1/role-training.json).
Add 28 agent-authored matched pairs covering all 14 CRediT roles.
Each pair keeps a project context and changes the task that determines
the role. Four pairs cover each of seven confusable role edges. Reject
normalized duplicates against the original training descriptions and all
evaluation wording banks. The [new cases](role-contrasts.json) are
synthetic and have no independent human adjudication. Matched edits are
an established direction in supervised language learning.
[Kaushik et al., ICLR 2020](https://iclr.cc/virtual_2020/poster_Sklgs0NFvr.html)

| Head | Examples | Objective |
| --- | --- | --- |
| Original CE | Original 84 | Multiclass cross-entropy |
| Augmented CE | Same 84 plus 56 new phrases | Multiclass cross-entropy |
| Shuffled pairs | Same 140 | Cross-entropy and a pairwise term. Rotate the right members within each role edge to break topic matching while preserving labels. |
| Matched pairs | Same 140 | Same objective, using the original topic-matched pairs |

For both pair-trained heads, the extra term rewards an increase in
the left-versus-right class logit gap when the text changes from
the right activity to the left activity. The term has weight 0.5.
Each fit uses a regularized ridge warm start on its training rows,
scaled by ten, then 120 full-batch AdamW steps at 0.002. Only the
linear weights train on CUDA. Each head uses top-one inference.
Use the original ridge scores and the unchanged 4B full menu as
references. All operating choices are fixed before evaluation.

## Results

A strict proof requires the exact intended source pair, a gold shared
author and both intended roles. Adverse counts include any issued proof
that fails those checks. The two panels contain 51 and 72 positives
and the same number of paired controls. Each cell is **strict / adverse**.

| Role classifier | Confirmation keywords | Extended keywords | Confirmation excerpts | Extended excerpts |
| --- | ---: | ---: | ---: | ---: |
| Original ridge | 24 / 10 | 30 / 5 | 30 / 14 | 44 / 8 |
| 4B full menu | 29 / 4 | 39 / 8 | 38 / 8 | 53 / 12 |
| Original CE | 30 / 4 | 37 / 1 | 36 / 8 | 52 / 4 |
| Augmented CE | 27 / 5 | 35 / 3 | 33 / 9 | 48 / 6 |
| Shuffled pairs | 27 / 5 | 35 / 3 | 33 / 9 | 48 / 6 |
| Matched pairs | 27 / 5 | 35 / 3 | 33 / 9 | 48 / 6 |

With the original training bank, cross-entropy improves the strict
count over ridge in both panels and both views and reduces the
observed adverse count. Against 4B, original CE yields eight fewer
adverse excerpt outputs in the extended panel with one fewer strict
proof. All four paired strict-outcome 95% intervals against 4B include
zero. This trial does not establish noninferiority. The role head
avoids the 4B forward at fixed read counts. The experiment does not
measure live latency or GPU throughput.

Adding these particular synthetic phrases lowers the strict count in
both views of both panels relative to original CE. Neither matched nor
shuffled pair supervision recovers the gap. When an intended paper is
withheld, original CE and 4B both produce zero keyword substitutions
and four or five excerpt substitutions, depending on the panel.

Test 28 frozen non-NISO engineering descriptions separately without
assembling source proofs. All students and ridge resolve 14/14 in
the original wording view and 13/14 in the alternate wording view.
The 4B route resolves 14/14 in both. This small check shows no
wording-level student gain. The main experiments still share a NISO
wording bank across source cohorts. Both comparisons concern known
role names and do not establish performance on unmapped duties.

Slurm array 46635 and job 46642 trained the heads on ws3 RTX 5090.
The four source tests reproduce in forward and reverse case order.
All five jobs generate identical weight hashes. The original
baseline results match the archived source comparison.
[Source-proof metrics](results.json) · [Wording metrics](wording-results.json)

## Reproduce

Use the [fixed-source menu reproduction inputs](../elife-fixed-pair-role-menu-v1#reproduce).
Run the original fixed-pair inference command in Slurm CUDA, replacing
`--menu-ablation` with
`--contrast-training benchmarks/elife-role-contrast-training-v1/role-contrasts.json`.
Repeat with `--reverse` for each panel, then run the same aggregation command.
For the wording check, use the same model revisions and input snapshots.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.role_contrast_wording_check QWEN_EMBEDDING QWEN_4B \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/paired-role-verification-v1/results.json \
  benchmarks/elife-role-contrast-training-v1/role-contrasts.json NEW_RESULTS
```
