# Separate contribution roles from additional proof requirements

Reuse the broad CRediT role menu and Qwen3-4B checkpoint from
[the taxonomy benchmark](../credit-taxonomy-transfer-v1).
Author 18 new three-way contrasts. Each contrast starts with a
single contribution role. Insert the same extra claim either as
mandatory or explicitly optional. Extras include measured outcomes,
source and project scope, operating conditions, attribution and
availability. The target in this test is permission to proceed
using the declaration alone. Work held for additional proof is
unresolved.

All routes require agreement on the contribution role in the
forward and reversed role menus. Compare that route alone, a
surface-pattern guard, a binary sufficiency prompt, and a prompt
that also identifies the kind of additional proof. The regex
guard uses simple lexical triggers on the entire input. The two
model guards each use opposite answer orderings. Record both
single decisions and the pair-agreement gate. Evaluate retention
on base and optional requests and bypasses on mandatory controls.
No source is read in this diagnostic. The timings add the
component decision times and exclude search and fallback.

A returned role on a mandatory case is tracked as a shortcut
bypass, even if the proposed role itself is correct. All requests
are authored and include explicit mandatory or optional wording.
This is a controlled stress test for the routing stage.

## Results

Both tasks in Slurm array 44201 completed on RTX 5090 and reproduced
all non-timing counts. Each condition has 18 requests. Every response
allowed through the guard has the correct normalized field except one
mandatory request accepted by the role-menu-only arm.

| Route | Base accepted | Optional accepted | Mandatory incorrectly allowed through |
| --- | ---: | ---: | ---: |
| Role menu alone | 18 | 18 | 18 |
| Surface pattern guard | 17 | 8 | 8 |
| Binary check, forward label order | 5 | 0 | 0 |
| Binary check, reverse label order | 1 | 0 | 0 |
| Binary check, both agree | 0 | 0 | 0 |
| Proof-type check, forward | 10 | 15 | 0 |
| Proof-type check, reverse | 6 | 3 | 0 |
| Proof-type check, both agree | 5 | 2 | 0 |

The strict gates suppress the shortcut failure by sending many
eligible requests for further proof. The four-choice forward check
retains more eligible cases than the strict agreement route, but
its behavior depends strongly on encoding. It is a post-hoc
observed operating point on an authored panel. A new test split
would be needed to select an encoding or set an acceptance target.
On optional-extra requests the summed local P95 is about 69 ms
for the role menus alone, 84 ms with the forward proof-type check,
and 99 ms with proof-type agreement. These are decision components;
they do not price any fallback reading.

[Both Slurm runs](results.json)

## Reproduce

Use an allocated Slurm GPU with the verified Qwen3-4B checkpoint.

```sh
export PYTHONPATH=src:.
python -m experiments.credit.study_credit_scope_gate NEW_RESULTS --order-offset 0
```

Repeat with offset 1 and a fresh output path.
