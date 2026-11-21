# One blinded review for recurring qualification disagreements

Combine the existing [prompt-scope](../enterprise-requirement-roles-v1),
[compact-recheck](../enterprise-compact-rechecks-v1) and
[direct qualification](../enterprise-qualification-scope-audit-v1)
packets. Verify the frozen source files and both the request and exact
clause coordinates against the original development projection. Collapse
only exact matches. Each reviewer sees the full request and marked
excerpt, with no model vote, method identifier or priority flag.

| Quantity | Count |
| --- | ---: |
| Input review rows | 167 |
| Distinct clause anchors | 154 |
| Distinct original requests | 46 |
| Anchors in two audits | 13 |
| Mandatory qualification disputed in one audit | 52 |
| Mandatory qualification disputed in two audits | 8 |
| Independent human judgments | 0 |

The private coordinator index prioritizes the eight clauses where two
audit settings each disagree on whether the excerpt is a mandatory
applicant qualification. This ranking depends on model disagreement.
It does not indicate which output is correct. The audits reuse data
and a model checkpoint, and the purposive panel does not estimate
population error. Use separate reviewer copies and adjudicate ambiguous
or conflicting roles. Preserve the exact original anchors and retain
the frozen aliases to avoid reviewing the same excerpt twice.

[Aggregate inventory](results.json)

## Reproduce

With authorized access to the original development projection and the
three linked private packets:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.merge_enterprise_scope_review \
  ORIGINAL_QUERIES ROLE_PACKET ROLE_TRACE COMPACT_PACKET COMPACT_TRACE \
  SCOPE_PACKET SCOPE_SIDECAR NEW_PRIVATE_PACKET
```

The output includes a validation manifest, two blank forms and a separate
private priority index. Keep the text, model linkage and submissions
outside the public repository.
