# Audit original sources before locking enterprise eligibility

The [public compendium](../enterprise-public-compendium-v4)
contains 530 provisional groups. This prospective pilot samples
65 distinct groups before any retrieval run or eligibility judgment.
The sampling unit is one frozen representative per group. Keep the
13 publication batches as strata, allocate two samples to each
batch, then distribute the remaining 39 by largest remainder using
the remaining stratum sizes. Order groups within each stratum by
the protocol's salted component digest. Reviewers see cases in
the order of a separate blinded case digest.

The parser reconstructs the entire frame from the original source
snapshots and prior-use records. It requires the pinned frame hash
and verifies all 530 selected representatives' text and source
positions against the original reconstruction before sampling.
The reviewer copies carry the local source batch and row locator,
the parsed technical title and body, and the issuer. Known contact
patterns are masked. The exact text hash and group exclusion key
are separate from the reviewer forms.

## Source-completeness finding

All 241 Xi'an representatives have a nonempty cooperation-requirements
field in the original attached form, including all 30 Xi'an cases in
the pilot. The frozen technical projection stops at that field.
Each Xi'an reviewer card displays the separately masked original
field in `unindexed_original_cooperation`. Compare it to the current
title and body during the preservation review. A nonempty
cooperation field is a candidate for review. It may contain an
administrative note, a preference, or a hard requirement. No
eligibility label is inferred from field presence.

## Review rubric

Open the original version for the indicated source locator first.
Independently assign each field `pass`, `fail`, or `uncertain`,
and write a rationale referring to the source location.

| Field | Pass condition |
| --- | --- |
| `source_and_issuer` | The identified original request and publishing or requesting organization are traceable. The locator binds to the same request. |
| `technical_scope` | The original request names a concrete technical problem or work objective and provides enough context to judge suitability. |
| `constraints_preserved` | The sampled text preserves the original task and explicit hard conditions, including relevant constraints in continuation or cooperation fields. An original with no explicit hard condition can pass. |
| `eligible_expert_search` | This is a technical need for which selecting a person is meaningful and source-groundable. A purely transactional purchasing notice or generic promotional text fails. |

Judge eligibility without system predictions or the other reviewer's
responses. Use the published version at its original timestamp for the
reference. Historicity is allowed; an active collaboration opportunity
would need a separate check. The checker classifies agreed pass,
agreed fail, missing reviews, and cases needing adjudication. An
uncertain field or field disagreement requires adjudication even if
both reviewers agree the case should ultimately be excluded.

The pilot can estimate source-specific eligibility after review.
For a pooled estimate, weight each stratum by its full-frame share,
since small batches are intentionally oversampled. The 65 pilot
groups are excluded from a later locked test and from related
organization or near-duplicate groups discovered by adjudication.
The remaining 465 are a provisional pool. This exercise does not
estimate expert-search precision, issuance, pair discordance, or
conditional error risk. Those require separately evaluated requests.

## Frozen pilot

Two hash seeds reproduced the frame, both reviewer copies and the
holdout exclusion map byte for byte. The validator requires the
original case inventory and rejects edits to source fields or review
copies with a shared reviewer ID. All 65 cases currently await both
reviews. [Aggregate receipt](results.json).

## Reproduce

Follow the original [compendium replay](../enterprise-public-compendium-v4#reproduce)
with fresh aggregate and private frame paths and append

```sh
--eligibility-pilot benchmarks/enterprise-eligibility-pilot-v1/protocol.json NEW_PRIVATE_PILOT
```

Keep both reviewer copies and the exclusion map local. After editing
only the judgment, reviewer and rationale fields, validate the forms:

```sh
PYTHONPATH=src:. python -m experiments.enterprise_eligibility_pilot \
  PRIVATE_PILOT REVIEWER_A_JSONL REVIEWER_B_JSONL
```
