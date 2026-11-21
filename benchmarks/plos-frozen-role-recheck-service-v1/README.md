# Test a frozen role recheck on new PLOS keyword requests

The [development replay](../plos-source-first-role-recheck-v1)
found three additional exact keyword proofs without additional
wrong outputs on the examined groups. Before constructing the
new evaluation, lock the source-score route, the Qwen3-4B
checkpoint, and the strict 4B logit margin of 26.125 from the
archived development result. Keep the eLife source scorer
and cutoff and the 0.6B ridge role decoder fixed. Fit nothing
on the new held groups.

## Independent paper groups

Use the [Europe PMC Articles API](https://dev.europepmc.org/RestfulWebService)
to identify 2024 PLOS ONE papers with an indexed ORCID and
machine-learning or optimization terms in the title or abstract.
Confirm an author and both contribution statements in publisher
JATS before selecting a pair. A valid matched contrast changes
one of the roles so no verified author satisfies both conditions.
Exclude earlier recorded PLOS and eLife publication links
by DOI and ORCID using the pinned archives. Every selected
pair is disjoint in recorded ORCIDs from the other new pairs.
Keep the whole old PLOS catalog as distractors and add the
selected sources for retrieval. The recorded identifiers do
not establish isolation for unrecorded authors or pretraining.
The freeze job uses no model outcomes.

## Request replay

Compare the vectorized exact baseline with the same baseline
plus the locked source-score 4B route. Only the latter tries a
recheck when the original role confidence blocks the query,
the independently proposed fused papers are distinct, and
the pair passes the frozen source score. Recheck only unclear
clauses together in one 4B forward, requiring that both cited
roles appear under the same author. Assert that the intervention
preserves every previously issued answer. Test the natural keyword
view and withdraw each intended paper separately. Score the
designated sources and requested roles. Keep counts of both
correct and wrong issued proofs, recheck work, and paired
effects clustered by publication group.

A single warmed local HTTP worker times the complete request
from client POST through JSON decoding. Encode each query
afresh, including repeated and withdrawn variants. Wall time
includes exact search, the 4B role check when used, and transport.
It excludes model and index initialization and represents
serial loopback execution.

## Results

The freeze job (47347) records 943 metadata entries, 61 pair
proposals, and 37 screened pairs. The 13 confirmed groups
contribute 26 new sources and 39 matched role edits. The resident
catalog contains 440 sources. Both tasks in array 47352 (ws6,
RTX 4090) reproduce every proof and abstention across opposite
request orders. No calibration uses the new groups.

| Condition | Policy | Correct / wrong | 4B calls | Client P50 (ms) | Client P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Intact, 78 requests | Baseline | 9 / 0 | 0 | 16.4–16.7 | 17.0–17.5 |
| Intact, 78 requests | Frozen 4B recheck | 10 / 0 | 47 | 50.5–50.6 | 79.4–80.1 |
| Withheld, 156 requests | Baseline | 0 / 1 | 0 | 16.4–16.6 | 16.8–17.2 |
| Withheld, 156 requests | Frozen 4B recheck | 0 / 1 | 54 | 16.8–17.1 | 76.4–77.4 |

The only additional proof occurs in one intact group.
The paired descriptive 95% interval is 0 to 0.231
additional correct proofs per group. The frozen route retains
the baseline's single substitute under withdrawal. Its 101
role forwards process 52,018 nonpadding input tokens across
both availability states. The effect establishes a cost
trade-off on a small independent paper benchmark. It does
not establish a new risk bound or an efficient served default.

[Publisher manifest](manifest.json) · [Results and digests](results.json)

## Reproduce

Restore the [source scoring inputs](../plos-contrastive-rescue-identity-v1#reproduce)
and download the 26 publisher XML files pinned in the new manifest
into a flat directory with exact file digests. Use the same
positional arguments as the scoring replay with the module
`experiments.credit.plos_frozen_role_recheck_service`.
Add `--fresh-manifest MANIFEST --fresh-xml XML_DIR --lock
benchmarks/plos-source-first-role-recheck-v1/results.json --reader
QWEN3_4B_INSTRUCT_2507`. Repeat with `--reverse`. The upstream
2024 data collector is
`experiments.credit.freeze_plos_role_recheck_holdout`; it needs
the archived earlier PLOS XML directories for identity screening.
