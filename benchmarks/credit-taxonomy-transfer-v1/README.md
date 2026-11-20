# Test a complete contribution menu on new phrasing and hard scope controls

Use the Qwen3-4B checkpoint from the
[PLOS role normalization benchmark](../plos-study-role-normalization-v1).
The new benchmark uses 56 freshly authored duty phrases, with three
per each of the 14 [CRediT roles](https://credit.niso.org/contributor-roles-defined/)
and 14 that require more than any single contribution declaration can
establish. Out-of-contract cases cover additional measurements,
qualifications, responsibility, or availability. Descriptions
in the menus paraphrase the publisher-independent taxonomy.

Compare a five-field menu corresponding to the preceding eLife
experiment with the 14-field menu. The five-field contract treats
the nine omitted fields as unavailable. Both contracts include an
explicit outside option. For each phrase, compare its standalone
form with a two-sentence form mentioning a different role as optional
before giving the actual requirement. The prompts use the same
wording with either menu. Forward and reversed choices use the
same permitted letter tokens. The agreement route maps a role only
when both choices return the same field.

The metric is exact role interpretation or correct abstention for
the chosen menu. The hard scope cases remain out of contract for
both menus. Runtime P95 covers local per-duty decision including
two independent serial calls for agreement. This test has no
article documents or entity labels. It measures query-side scope
sensitivity, with synthetic phrasing and small per-role counts.

## Results

Both Slurm tasks in array 44191 completed on RTX 5090 and matched
every outcome count. The table shows the agreement route. For the
five-field contract, the additional-field column counts correct
abstentions. For the full contract, it counts correct mappings.

| Menu | Wording | Original roles / 15 | Additional roles / 27 | Hard scope abstentions / 14 | P95, ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Five | Standalone | 14 | 12 | 5 | 41.30–41.42 |
| Five | Optional distractor | 14 | 17 | 6 | 41.77–41.90 |
| Fourteen | Standalone | 15 | 26 | 1 | 59.23–59.34 |
| Fourteen | Optional distractor | 15 | 27 | 1 | 69.96–70.16 |

The full menu covers valid research contribution roles that the
five-field contract omits. It leaves nearly all hard proof conditions
unchecked. Even agreement between two encodings often maps the
requirement to a plausible field. A role normalization result should
therefore preserve additional hard constraints for separate source
verification. The evaluation uses authored cases and has no evidence
about their prevalence or the downstream cost of the added checks.

[Both Slurm runs](results.json)

## Reproduce

Use an allocated Slurm GPU with the verified Qwen3-4B checkpoint.

```sh
export PYTHONPATH=src:.
python -m experiments.credit.study_credit_taxonomy_transfer NEW_RESULTS --order-offset 0
```

Repeat with order offset 1 and a fresh result path.
