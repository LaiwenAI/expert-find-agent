# Bound topical-support gains before reviewing gap-refill citations

Replay the [four-source gap-refill experiment](../role-gap-title-refill-v1)
with an optional exact citation trace. Job `44063` on ws5 reproduces
all non-timing aggregates from both frozen runs. Audit the three
four-source methods at full available declaration visibility. Compare
title refill against role-eligible hybrid refill and the fixed
role-aware two-view mix.

Deduplicate identical topic–article pairs. Recheck emitted and
tagged slot counts from the trace against the replay. Reviewers see
the exact title and abstract and the requested topic. They do not
see DOI, route, labels, rank or priority. The assigner retains the
priority list privately. A supported case requires a direct topical
connection in the visible text. Publisher declarations bind roles
to people. This review addresses topical support in an article,
with enterprise task fit and claims of individual responsibility
requiring separate evidence.

## Results before adjudication

No human labels were available. The two blank forms contain 51
PLOS and 143 eLife cases. Reconciliation verifies the immutable
views and keeps all 194 cases unresolved.

| Contrast, title refill minus comparator | Supported-slot difference | Title-refill / comparator slots issued | Influential cases for supported slots | Minimum decisive cases for a range at most 2 pp |
| --- | ---: | ---: | ---: | ---: |
| PLOS vs role-eligible hybrid refill | −14 to +5 | 59 / 68 | 12 | 8 |
| PLOS vs fixed two-view mix | −4 to +6 | 59 / 57 | 8 | 4 |
| eLife vs role-eligible hybrid refill | −34 to +24 | 245 / 255 | 30 | 21 |
| eLife vs fixed two-view mix | −35 to +42 | 245 / 238 | 47 | 36 |

In eLife's comparison with hybrid refill, the supported fraction
among recommendations issued has a difference range of −13.33 to
+13.33 points. Tightening that range to a width of two points takes
at least 51 decisive case judgments even under optimal ordering.
The two metrics answer different questions. Supported slots use five
positions per request. Conditional support divides by the number
actually issued, so cases on which both systems agree can still
inform the comparison.

For each metric, a case has a signed coefficient equal to its
difference in citation counts or normalized fractions. The exact
bounds assign each unresolved coefficient adversarially. Resolving
one case shrinks interval width by the absolute coefficient,
regardless of the label. Sorting by absolute coefficient minimizes
the number of decisive cases required for a width target for one
contrast. This is a finite-cohort bound with no sampling confidence
claim. The two-point target guides review effort and does not
guarantee a positive lower bound or determine a winner.

As sensitivity only, assuming every tagged case is supported
narrows the eLife slot difference versus hybrid refill to
−20 through +7. That range still includes both directions.
Independent review is needed before treating the higher major-topic
proxy count as a gain in supported recommendations.

[Aggregate audit](results.json)

## Reproduce

Run the gap-refill replay with `--witness-output NEW_WITNESSES`.
Use the same projections and the pinned gap-refill reference.
Keep the assigner packet and completed reviewer forms private.
Independent reviewers each supply relevance and a short rationale.
Unclear judgments and disagreements remain unresolved.

```sh
PYTHONPATH=src:. python -m experiments.credit.role_gap_review prepare \
  PLOS_PROJECTION ELIFE_PROJECTION MESH_SNAPSHOT REPLAY NEW_WITNESSES \
  benchmarks/role-gap-title-refill-v1/results.json NEW_PRIVATE_REVIEW_DIR NEW_SUMMARY
PYTHONPATH=src:. python -m experiments.credit.role_gap_review reconcile \
  NEW_PRIVATE_REVIEW_DIR/packet.json REVIEW_A REVIEW_B NEW_RECONCILIATION
```

## First review batch

Create a focused assignment from the existing private review packet.
For these frozen inputs, the first batch contains the 21 most
influential cases for the eLife supported-slot contrast against
role-eligible hybrid refill. If both reviewers provide decisive
matching judgments for all 21, the remaining identification range
has a width of 1.82 percentage points. The location of that interval
depends on the judgments. All 173 unassigned cases remain unresolved.

The two ZIPs contain separate blank forms, instructions, and a
standalone offline review page. The review page displays the
topic and exact title and abstract, accepts a rationale, and exports
a JSONL file that can be reloaded to resume. It makes no network
requests. The organizer retains `assignment.json` and the full
packet. Neither is needed by a reviewer.

```sh
PYTHONPATH=src:. python -m experiments.credit.role_gap_review assign \
  PRIVATE_REVIEW_DIR/packet.json benchmarks/role-gap-review-v1/offline-review.html NEW_BATCH_DIR
PYTHONPATH=src:. python -m experiments.credit.role_gap_review reconcile \
  PRIVATE_REVIEW_DIR/packet.json REVIEW_A_EXPORT REVIEW_B_EXPORT NEW_RECONCILIATION \
  --assignment NEW_BATCH_DIR/assignment.json
```

The reconciliation checks the packet fingerprint, requires the
exact case set, and checks every exported view against the frozen
text. The organizer must establish reviewer independence. A partial
export is valid when the untouched cases remain present in the form.
