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

## First review batch

Use the eight anchors with repeated mandatory-qualification disagreement.
For each, choose one distinct comparison anchor without a within-audit
mandatory-qualification split. Prefer the same original request, then
closest excerpt and request length. In this inventory all eight pairs
come from the same request within each pair. The reviewers see a single
deterministically shuffled form with no pair or stratum labels. The
selected panel is an adjudication aid and does not support an accuracy
or representativeness claim. [Batch inventory](first-batch.json)

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_scope_review_batch prepare \
  PRIVATE_PACKET PRIVATE_BATCH
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_scope_review_batch transfer \
  PRIVATE_PACKET PRIVATE_BATCH RETURNED_A RETURNED_B NEW_PRIVATE_SUBMISSIONS
```

Use distinct reviewer IDs. A blank case can remain blank. The transfer
validates the frozen source and copies only the judgment fields. Pass
`--master-a` and `--master-b` to preserve previously completed judgments
from the larger packet. Conflicting judgments for the same reviewer and
anchor require reconciliation before transfer. The returned full-size
copies can go through the existing two-reviewer validation.
