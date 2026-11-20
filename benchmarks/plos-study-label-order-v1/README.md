# Test binary answer encoding on authentic PLOS contribution notes

Reuse the exact 59 matched pairs from the
[same-initials stress test](../plos-study-bound-initials-v1).
Keep the verifier checkpoint, identifying report alias, task roles
and evidence scope fixed. Compare two equivalent instructions
that swap which output token means yes. Also calculate conjunction,
which accepts only if both encodings accept. The task prefix in both
encodings differs slightly from the one used in the original study.
For comparisons within this experiment, the only prompt difference
is the association between the label tokens and yes/no.

Slurm array 44132 completed both tasks on an RTX 5090 and reproduced
each decision count. Each positive or negative column has 59 cases.
Pair correctness requires the positive and negative from the same
pair to both be correct.

| Read scope | Yes token | Supported accepted | Unsupported accepted | Both correct |
| --- | --- | ---: | ---: | ---: |
| Both reports | 1 | 57 | 43 | 16 |
| Both reports | 0 | 9 | 2 | 9 |
| Named report | 1 | 59 | 52 | 7 |
| Named report | 0 | 38 | 15 | 25 |
| Named report and matching initials clauses | 1 | 59 | 56 | 3 |
| Named report and matching initials clauses | 0 | 42 | 33 | 18 |

Conjunction preserves the yes-token-0 counts for the first two scopes.
For scoped clauses it accepts 42 positives and 32 negatives. Named
report conjunction takes 31.0–31.2 ms local P95 versus 15.5–15.6 ms
for one call. This is a selected stress panel, so the counts do not
estimate open-search prevalence. Label sensitivity is substantial
even when report identity is given. Conjunction adds cost and provides
little improvement over the more conservative encoding. Choose and
validate a fixed output encoding in subsequent experiments, and
test independent role-by-role evidence on expressed constraints.

[Both Slurm runs](results.json)

## Reproduce

Restore the original pinned publisher captures. Use an allocated
Slurm GPU with the Qwen3-4B checkpoint required by the verifier.

```sh
export PYTHONPATH=src:.
python -m experiments.credit.study_bound_initials_label_order build NEW_FIXTURE \
  LEGACY_SNAPSHOT INDEPENDENT_SNAPSHOT BODY_TRANSFER_SNAPSHOT
python -m experiments.credit.study_bound_initials_label_order run NEW_FIXTURE NEW_RESULTS
```
