# Transfer frozen original-text review to new publication groups

Reuse the index, fusion, and selective review policies from the
[budget-matched comparison](../selective-conjunction-review-v1)
without fitting any parameters on PLOS. The verifier thresholds,
development routing strata and cutoffs, router weights, and
eLife and Sensors aggregate checks must reproduce before the PLOS
outcomes are accepted.

Build the test from archived PLOS ONE JATS snapshots. Expand the
inventory from bounded candidate lookups on up to 48 seed authors.
Use the [PLOS Search API](https://api.plos.org/solr/search-fields/)
for exact names, and [Europe PMC](https://europepmc.org/RestfulWebService)
for ORCID searches. Select anchors from cached articles not already
chosen in the preliminary linked panel. Inspect up to three distinct new documents per search
from the bounded response and date window. A new paper is usable
only if its publisher XML confirms the link by checksum-valid ORCID.
Freeze the successful XML object hashes and search receipts. Skip
conflicting documents, unsupported licenses and missing roles for
linked contributors. Exclude overlap with the recorded, resolved
authors in the two prior eLife proof panels and the linked
engineering publication pool.

Within each PLOS panel, groups are source- and recorded-author-disjoint.
One panel keeps source swaps. A second panel changes only one of
the two requested roles, using a role declared on that same paper.
Both designs require that a common author support the positive and
that no common author satisfy the negative. Use up to three matched
families per group. The two panels share some source groups and are
reported separately. The cached five-group swap pilot is the intake
check and is omitted from the transfer totals.

The model sees the actual JATS role strings in their original field
order, attached to numeric author aliases. These fields are wordings
of a controlled taxonomy, with no general prose entailment claim.
The role labels identify what the publisher declares. The experiment
uses the existing NISO query phrase bank; phrase templates can
recur across publisher panels. Source selection and author linkage
are provided.

Compare the pure index and full fusion with the original fixed
routing policies. The matched replay holds constant the number of
requests and original-text clause checks in each cost stratum.
The per-request replay uses the thresholds frozen on eLife and
measures actual verifier work and local component P95. Raw context
and author memberships are prepared before the timer. The measured
path does not include network, queueing, or open-corpus retrieval.

## Results

The two discovery routes verify two distinct new XML documents in
total. PLOS Search proposes 50 distinct documents and Europe PMC
proposes 27. The combined admitted inventory contains 160 papers.
Freeze six source-swap groups (14 positives and 14 negatives) and
nine role-edit groups (27 positives and 27 negatives). Five
publication pairs appear in both panels. Because phrase selection is
keyed to the frozen panel, the five-group pilot is a pipeline check
rather than a paired count comparison against the expanded panel.

| Live route | Source swaps correct / wrong | Raw checks | P95 (ms) | Role edits correct / wrong | Raw checks | P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Index only | 13 / 0 | 0 | 10.93–10.97 | 23 / 1 | 0 | 10.78–10.81 |
| Margin | 13 / 0 | 10 | 20.01–20.36 | 19 / 1 | 26 | 20.37–21.36 |
| Disagreement | 13 / 0 | 14 | 20.49–20.75 | 21 / 1 | 28 | 20.42–20.74 |
| Disagreement veto | 13 / 0 | 14 | 20.44–20.70 | 21 / 1 | 28 | 20.31–21.19 |
| Learned gain | 13 / 0 | 32 | 20.62–21.41 | 23 / 1 | 46 | 20.38–20.70 |
| Full fusion | 13 / 0 | 80 | 21.06–21.37 | 19 / 1 | 144 | 21.27–21.36 |

In the matched replay, all selective routes use 20 checks on
source swaps and 34 on role edits. They all tie the index on
source swaps. Hash and learned gain tie the index on role edits
(23 correct, one wrong), margin returns 19 correct and one wrong,
and disagreement returns 21 correct and one wrong. None reduces
the observed number of wrong requests. The earlier one-error
Sensors reduction does not replicate on these small PLOS panels.
On role edits, the veto loses two known positives and keeps one
wrong request. It spends fewer original-text checks than fusion,
while leaving the slow path in the client-visible component tail.
The results support retaining the index as the current PLOS
control. They do not settle selective review on unrestricted
prose, or on collections with noisier linking.

Slurm arrays 47062 and 47072 on ws1 (RTX 4090) reproduce routing,
work, and response signatures in both traversal orders. The
[discovery receipts](source-intake.json) and the manifests below
publish only article identifiers, hashes and role contrasts.
[Aggregate results](results.json).

## Reproduce

Use the [source-swap manifest](combined-swaps-manifest.json) or the
[role-edit manifest](role-edits-plos-manifest.json). Restore the
pinned PLOS ONE XML documents from the publisher by DOI and use
the original eLife notes and engineering panel for the identity
exclusion. The restorer accepts a folder containing the matching
raw XML files.

```sh
PYTHONPATH=src:. python -m experiments.credit.plos_selective_review_holdout \
  ELIFE_NOTES CROSS_SOURCE_PANEL NEW_MANIFEST NEW_NOTES PLOS_XML_DIRECTORY \
  --restore FROZEN_PLOS_MANIFEST
```

Run the [selective review runner](../selective-conjunction-review-v1)
with its original positional inputs and add the following options.
Use a Slurm GPU and repeat with `--reverse`.

```sh
--external-manifest NEW_MANIFEST --external-notes NEW_NOTES \
--router-archive benchmarks/selective-conjunction-review-v1/results.json
```
