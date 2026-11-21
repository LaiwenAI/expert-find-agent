# Freeze and evaluate a larger publisher-declared role holdout

Use the pinned [eLife XML release](https://github.com/elifesciences/elife-article-xml/tree/58250dd64e27afae6bc2700bcddd84dd38776e0d/articles).
Select 2,400 stable hash-ranked latest-version article IDs from the same
modern article band as the [previous holdout](../elife-joint-postings-service-v1).
Remove all IDs in both preceding prospective article samples before selection.
Recreate the complete earlier exposure inventory, check its archived hashes
and verify each previous XML object's Git and SHA-256 digests. Exclude sampled
articles sharing a title, DOI, valid ORCID or normalized author name with
an examined source. Reject unsuitable document types, unsupported licenses
and incomplete role or abstract projections. Pin all retained and excluded
article hashes before model evaluation.

For each linked paper pair, construct one author-role conjunction and
one matched role-swap query. Use both a keyword and an excerpt view of
each query. Use the existing one-family-per-paper-pair rule and pair
the quality intervals by connected paper and recorded-author component.
Source text includes the title and first abstract portion. Query clues
use the held portion. The task measures agreement with publisher-declared
roles and exact source identity.

Freeze the existing single-slot baseline, lightweight cascade, dense
source fallback, and calibrated resident source gate with 4B role checks.
Recompute the previous gate calibration from the archived inputs and
require exact agreement with its published reference. No held labels
choose a threshold. Each search uses at most one verifier forward.
Separately hide each intended paper once, preserving the query and the
rest of the index. The three cascade policies must abstain when they
cannot establish the exact requested proof. Report substitute issuance
and verification calls separately. The single-slot baseline is only
measured with the full catalog because its original API has no mask.

Repeat in both request and arm orders. Time warmed serial execution from
the structured query to its evidence response. Count strict proof and
adverse issuance with the catalog intact. Report missing-source outputs
on the fixed stress cases and use connected paper and author groups, rather than repeated
views or deletions, as the quality sampling unit. The measured workload
has structured inputs and no independent review of enterprise suitability.

## Results

The two independent builds reproduce byte-identical projections and
manifests. Of 2,400 sampled IDs, author exposure excludes 1,060,
article type excludes 281, and missing author or heading metadata
excludes two. Forty-seven more records fail the complete proof-index
requirements. The resulting 1,010 indexed articles yield 72 linked
paper pairs and 55 connected paper and author groups. Slurm array
46431 evaluates 288 structured requests on ws3 RTX 5090. Both request
and arm orders reproduce identical evidence responses in the intact
and withheld catalogs. The format rule agrees with all query views.

| Complete catalog | Keyword strict / adverse / wrong-source | Excerpt strict / adverse | Keyword serial P95 (ms) |
| --- | ---: | ---: | ---: |
| Single-slot baseline | 39 / 0 / 0 | 45 / 0 | 27.7–27.8 |
| Lightweight cascade | 41 / 1 / 0 | 53 / 1 | 28.0–28.3 |
| Dense source fallback | 45 / 5 / 4 | 53 / 1 | 27.9–28.4 |
| Calibrated source gate plus 4B role check | 46 / 4 / 3 | 53 / 1 | 29.2–29.6 |

There are 72 positives and 72 role-swap controls per view. The
calibrated gate gains five strict keyword proofs relative to the
cascade, with a descriptive grouped interval of +1.5 to +12.7
percentage points among positives, and adds three adverse issuances.
Adverse issuances affect four of its 41 issuing keyword groups,
versus one of 37 for the cascade. The gate produces no excerpt gains.
Compared with the dense fallback, it trades 12 keyword source forwards
for nine role forwards and excludes one additional wrong-source result.

| One intended source withheld | Keyword substitute issuances / 288 | Keyword groups affected / 55 | Excerpt substitute issuances / 288 |
| --- | ---: | ---: | ---: |
| Lightweight cascade | 1 | 1 | 6 |
| Dense source fallback | 6 | 3 | 6 |
| Calibrated source gate plus 4B role check | 16 | 6 | 6 |

Neither removal of an intended source nor the repeated run increases
the number of independent statistical units. The 16 keyword substitute
outputs are all missing the exact requested proof. These outputs are
failures of exact evidence recovery and can refer to another topic or
another evidenced role. The test does not certify suitability for an
enterprise task. The earlier keyword calibration used only three wrong
source pairs. These results establish a quality and risk trade-off on a
larger fresh catalog. They support leaving the calibrated fallback
experimental and using the lightweight cascade as the shared reference.

[Frozen manifest](manifest.json) · [Aggregate results](results.json)

## Reproduce

Get the pinned article tree and restore the four earlier XML directories
as described in the [prior holdout](../elife-joint-postings-service-v1#reproduce).
Supply both preceding prospective manifests. The builder can fetch the
three additional frozen sets into their designated folders.

```sh
PYTHONPATH=src:. python -m experiments.credit.elife_extended_role_holdout \
  PUBLISHER_TREE PREVIOUS_MANIFEST LATEST_MANIFEST PREVIOUS_XML LATEST_XML \
  FRESH_XML FRESH_PROJECTION FRESH_MANIFEST \
  EARLIER_ELIFE_XML EARLIER_PROMPT_XML ENGINEERING_DUAL_XML ENGINEERING_LINKED_XML --fetch
```

Use the [matched-service command](../elife-matched-source-service-v1#reproduce)
on a Slurm GPU. Use the new manifest and projection and replace
`--matched-source-service-reference` with `--fresh-source-reference`.
The reference must contain the original `matched_calibration` object.
Repeat with `--reverse` and a fresh output.
