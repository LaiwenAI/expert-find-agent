# Independently test the frozen source-identity guard

Freeze the exact source guard from the
[prior diagnostic](../elife-source-identity-audit-v1). Both candidate
papers must be in the fused top three for their respective clues.
Each clue must have higher cosine similarity to its own paper than
to the other proposed paper. Apply this test before the existing 4B
role check without changing cutoffs, source scores, or reader prompts.

Use the same pinned publisher release and reconstruct the full
[prior exposure chain](../elife-extended-role-holdout-v1).
Exclude any earlier article ID before selection. Take all 365 unexamined
modern article IDs and the first 2,335 earlier IDs in a fixed hash
order. The age mix is intentionally different from the diagnostic.
Verify every archived and newly sampled Git object. Exclude shared
DOIs, normalized titles, normalized author names and recorded ORCIDs
before selecting any query family. Freeze the manifest before running
a GPU job. Independence refers to the experiment's source and recorded
author inventory. It does not establish isolation from model pretraining.

Use both forms of each held paper-pair query, including matched role-swap
controls. The keyword view is the primary comparison. Measure the change
in strict proof against the lightweight cascade and retain the complete-
catalog and withheld-source adverse counts. The excerpt view and the
original dense and calibrated fallbacks are controls. In the stress
test, hide each intended source separately and preserve the original
query. A substitute result does not establish the exact requested
proof. Report conditional adverse counts at connected paper and
recorded-author group level alongside request counts. Evaluate each
request and arm order twice with a single-forward ceiling. Report warm
serial P95 separately from quality.

## Results

Two independent corpus builds reproduce the manifest and projection
byte for byte. Of 2,700 sampled IDs, 1,506 are excluded for recorded
author overlap, 235 for article type, and five for missing metadata.
Another 70 fail the complete proof-index requirements. The resulting
884 indexed articles yield 51 linked paper pairs in 46 connected
paper and author groups. Slurm array 46455 completes both orders on
ws3 RTX 5090. Every issued response and every abstention reproduce
exactly across orders, including the source-withholding cases.

| Complete catalog policy | Keyword strict / 51 | Keyword adverse / 102 | Excerpt strict / 51 | Excerpt adverse / 102 | Keyword serial P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-slot baseline | 29 | 1 | 35 | 0 | 24.5–26.3 |
| Lightweight cascade | 29 | 1 | 37 | 0 | 26.9–27.8 |
| Dense source fallback | 32 | 2 | 38 | 0 | 26.3–26.8 |
| Calibrated fallback | 32 | 2 | 37 | 0 | 27.7–28.7 |
| Frozen source-identity guard | 31 | 1 | 37 | 0 | 27.5–28.2 |

| One intended paper withheld | Keyword substitute issuance / 204 | Keyword affected groups / 46 | Excerpt substitute issuance / 204 |
| --- | ---: | ---: | ---: |
| Lightweight cascade | 0 | 0 | 3 |
| Dense source fallback | 4 | 1 | 3 |
| Calibrated fallback | 9 | 3 | 3 |
| Frozen source-identity guard | 0 | 0 | 3 |

The frozen guard retains two of the three new strict keyword proofs.
It reproduces the cascade's adverse counts in both availability
conditions. In the intact keyword view, one of its 31 issuing groups
has an adverse output, compared with one of the cascade's 30 issuing
groups. It increases intact keyword role forwards from 20 to 22 while
keeping source forwards at 12. The grouped descriptive interval for
the strict improvement among positive keyword requests is 0 to +9.8
percentage points. The bootstrap uses the 46 connected groups. The
helper's generic sampling-unit text in the raw JSON refers to its
default use and does not spell out the grouping supplied here.

The result prospectively reproduces the direction and observed risk
reduction of the preceding diagnostic. Sample size and the interval
leave the size of the improvement uncertain. The conditional results
concern exact publisher-declared evidence on structured inputs.
They do not establish expert suitability for an enterprise request,
independence from training data, or end-to-end production cost.

[Frozen source manifest](manifest.json) · [All runs and grouped intervals](results.json)

## Reproduce

Use the [previous cohort builder](../elife-extended-role-holdout-v1#reproduce)
with the same publisher tree, archived manifests, and earlier XML
directories. Add `--extended EXTENDED_MANIFEST --extended-xml EXTENDED_XML`.
Use distinct output paths for the new projection and manifest.

Run the [frozen source-guard comparison](../elife-source-identity-audit-v1#reproduce)
with the new projection and manifest, `--fresh-source-reference`,
and `--guarded-source-replay`. Repeat with `--reverse` and a fresh output.
