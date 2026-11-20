# Test automatic role anchors before allowing declaration-only checks

Build on the [residual packet](../credit-residual-packets-v1)
experiment. The same 18 contribution and extra-proof pairs yield
nine layouts each. Add a clause that embeds the extra requirement
inside a relative clause without explicit mandatory wording.
The program can no longer see the known standalone role span.
Qwen3-4B must point to a complete original clause or decline.

Compare one choice menu, agreement after reversing the menu,
agreement plus a local lexical guard, and agreement plus a local
semantic proof check run in two answer orders. The lexical guard
uses the prior surface pattern and generic join and condition
markers. Every route retains unmarked clauses as unresolved.
Count both an incorrect role anchor and a full bypass of the
residual-proof path. A complete source anchor verifies the
quotation only. Correct anchoring also requires that the selected
clause state only one supported contribution.

The secondary panel reuses 48 bilingual enterprise controls.
Their 108 annotated qualifications, future-work and operating
requirements must stay in unresolved spans. No role-only anchors
are annotated for those requests. This check is conservative and
does not measure whether every proposed role within a full demand
is semantically correct. All input text in this replay is public
or authored. The runtime sums local anchor/proof decisions and
small preprocessing costs; full search is outside this test.

## Results

Both tasks in Slurm array 44206 completed on an RTX 5090.
All outcome counts match. The two attached layouts contain 36
mandatory cases. The remaining constructed mandatory layouts
have a separate residual clause and none bypasses all proof
work in any route.

| Route | Standalone permitted / 18 | Explicit preference permitted / 18 | Attached mandatory bypass / 36 | Bilingual requirements retained / 108 |
| --- | ---: | ---: | ---: | ---: |
| One choice menu | 18 | 15 | 36 | 71 |
| Two choices agree | 10 | 15 | 12 | 83 |
| Agreement with lexical guard | 8 | 13 | 0 | 83 |
| Agreement with local semantic proof check | 0 | 0 | 0 | 108 |

The lexical guard blocks the constructed attached-clause bypasses.
It still discharges 25 annotated requirements from the unresolved
inventory on the independent bilingual controls. Requests in that
panel keep at least one unresolved clause, so none qualifies for
the role-only path. This distinguishes unsafe clause removal
from allowing the whole request to bypass proof.
The additional semantic check retains every annotated requirement
and also blocks all standalone role-only controls. These results
support using automatic role anchors for nomination while
retaining all clauses until independently verified. They do not
establish a deployment operating point for a semantic scope guard.

[Both GPU runs](results.json)

## Reproduce

Use the pinned checkpoint in a Slurm GPU allocation.

```sh
export PYTHONPATH=src:.
python -m experiments.credit.study_credit_role_anchors \
  benchmarks/enterprise-requirement-roles-v1/controls.json NEW_RESULTS --order-offset 0
```

Repeat with order offset 1 and a fresh result path.
