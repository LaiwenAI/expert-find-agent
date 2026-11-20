# Preserve unverified clauses after a trusted role interpretation

Build a requirement packet from the complete source request and,
when available, one trusted verbatim span already verified to
require only a contribution role. Keep every other clause as an
unresolved source interval unless it begins with an explicit,
whole-clause optional marker and contains no recognized mandatory
cue. A role anchor must exactly match one clause. Offsets and
verbatim text are preserved. A missing role anchor leaves the
request unresolved.

The builder assumes an external verifier supplied a valid role-only
span. A whole-clause offset match checks integrity, not semantic
sufficiency. In a clause that combines a role and a harder
requirement, the caller must leave the role anchor unresolved.
This utility is evaluated separately from automatic extraction,
evidence search and the final eligibility decision.

## Controlled replay

Reuse the 18 authored pairs from the
[scope gate benchmark](../credit-role-proof-gate-v1).
Test eight layouts, supplying the known standalone role span
when the controlled request has one. The attached mandatory
condition has no separate role-only span. For comparison, apply
the previous regex to the complete request.

| Layout | Packet permits role-only / 18 | Whole-request regex permits role-only / 18 |
| --- | ---: | ---: |
| Standalone role | 18 | 17 |
| Mandatory extension, trailing | 0 | 8 |
| Mandatory extension, leading | 0 | 8 |
| Mandatory extension inside the role clause | 0 | 8 |
| Explicit whole-clause preference | 17 | 8 |
| Unrecognized preference wording | 0 | 8 |
| Preference followed by a mandatory clause | 0 | 8 |
| Both cues in the same clause | 0 | 8 |

The explicit preference case that remains unresolved contains a
mandatory-looking phrase within the optional sentence. The
conservative lexical check leaves it for review. These outcomes
use supplied role spans and templated cues.

On 48 separately released bilingual enterprise controls, all
144 annotated intervals remain represented in unresolved clauses.
None has an externally verified role-only anchor, so none enters
the role-only path. This validates source preservation rather
than classification of the 144 spans. A separate randomized
check confirmed source partitioning on 1,000 mixed whitespace and
punctuation inputs. Both CPU replays give the same counts and
the largest observed per-layout timing P95 is under 13 microseconds.

[Aggregate results](results.json)

## Reproduce

```sh
PYTHONPATH=src:. python -m experiments.credit.study_credit_residual_packets \
  benchmarks/enterprise-requirement-roles-v1/controls.json NEW_RESULTS
```
