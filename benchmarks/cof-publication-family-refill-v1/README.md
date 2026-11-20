# Replace duplicate publication versions within the CoF source budget

Freeze the two [full-catalog CoF panels](../cof-merit-resident-bundles-v1)
and the dense best-source reviewer ranking. Each arm proposes
up to three owned records per expert. Apply the same frozen
verifier and two-positive rule. In the adjusted arms the
selected records belong to distinct groups under the stated
identity definition, so a single group cannot fill both slots.

| Arm | Source identity | Candidate records available per expert |
| --- | --- | ---: |
| Raw | Individual indexed record IDs | 3 |
| Version prefix | Two externally confirmed version families | 3 |
| Version refill | Same confirmed families | 12 |
| Title prefix | Unicode-normalized title equality | 3 |
| Title refill | Same title proxy | 12 |

Keep the highest-ranked available record from each group.
The refill arms continue down the cosine order until they
find three different groups or exhaust the twelve records.
They do not add verifier slots. The confirmed families are
the conference and journal versions manually identified in
the [depth review](../cof-profile-depth-review-v1).
They came from these evaluation panels, so the version arms
are targeted descriptive replays. Title-only grouping is a
sensitivity control. It can miss renamed versions or merge
different works that share a title. Distinct groups under
either rule need not constitute independent contributions.

Measure candidate and citation-view changes, raw citations,
citations without identified duplicate groups, known-positive
and known-negative counts, and warm P95 from query text through
verification. Require every arm to preserve the incoming
reviewer order for all queries. Rotate the arm order. Report
unjudged slots separately. These sparse reviewer labels and
model judgments do not count as human source validation.

## Results

Slurm array 45343 completed both panels on an RTX 5090.
All 181 requests preserve the incoming reviewer order in all
five arms. The original dense route reproduces the published
cited-positive counts. Slurm job 45346 passes both GPU index
tests, including the wider owned-source limit with and without
an eligible-owner restriction.

| Panel | Arm | Cited slots | Cited known positives | Known positives with distinct titles | Known positives without a confirmed version duplicate | Complete warm P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KDD | Raw | 264 | 13 | 6 | 8 | 126.67 ms |
| KDD | Version prefix | 262 | 11 | 9 | 11 | 126.51 ms |
| KDD | Version refill | 262 | 11 | 9 | 11 | 126.37 ms |
| KDD | Title prefix | 184 | 9 | 9 | 9 | 122.81 ms |
| KDD | Title refill | 195 | 9 | 9 | 9 | 133.12 ms |
| SciRepEval | Raw | 93 | 25 | 24 | 25 | 122.38 ms |
| SciRepEval | Version prefix | 93 | 25 | 24 | 25 | 121.97 ms |
| SciRepEval | Version refill | 93 | 25 | 24 | 25 | 122.15 ms |
| SciRepEval | Title prefix | 89 | 24 | 24 | 24 | 121.40 ms |
| SciRepEval | Title refill | 90 | 24 | 24 | 24 | 122.54 ms |

KDD has four records in the confirmed families and five
raw citation packages that count both versions of a family.
Applying the targeted rule removes all five duplicate
packages. Cited slots fall by two, with a net increase of
three known positives under the same version-aware metric.
No confirmed family member occurs in the SciRepEval
catalog. The resulting version arms there are execution
controls for the same policy on unaffected inputs.

Title-only refill increases cited coverage relative to
title-only prefix, at the same known-positive total on
both panels. On KDD it adds two cited judged negatives
and nine cited unjudged slots, with observed P95 about
10 ms higher. Relative to raw, both identity-aware
refill variants have paired intervals touching zero
for the distinct-title cited-positive change on KDD.
The title-only rule changes 258 of 610 input source
lists on KDD and eight of 295 on SciRepEval.

The two confirmed families each consist of a conference
and a journal DOI. The four [Crossref work records](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)
have empty structured relationship fields. The version
link in this replay comes from author documentation.
The replay supports a targeted identity guard. It leaves
automatic version linking, budgeted replacement gains
and independent evidence validity open.

[KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json)

## Reproduce

Use the same pinned checkpoints and input snapshots as the
[full-catalog MERIT benchmark](../cof-merit-resident-bundles-v1).
Run on a Slurm CUDA allocation with a fresh result path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_publication_family_refill \
  KDD QWEN06B_CHECKPOINT RERANKER_CHECKPOINT KDD_BASELINE_JSON NEW_RESULTS
```

Use `SciRepEval` and its baseline file for the second panel.
