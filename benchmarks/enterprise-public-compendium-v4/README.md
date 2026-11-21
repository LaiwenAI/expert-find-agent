# Extend the provisional enterprise frame to 530 groups

Use two populated public demand compilations with explicit issuers.
The [2023 Yangzhou compilation](https://kyc.sues.edu.cn/_upload/article/files/36/7f/ee40e6944e30a92e6d233ce87d43/09e0d7fe-7809-46fd-8030-edc174d17a47.pdf)
was compiled by the local science and technology bureau and hosted by
Shanghai University of Engineering Science. The
[2025 Fujian compilation](https://www.qztc.edu.cn/_upload/article/files/14/41/8a58b053438abf97c4bf591c2a59/d28bb580-e25e-478e-ad04-f5f347560d62.pdf)
is hosted by Quanzhou Normal University.

The Yangzhou parser recovers 206 forms from the 439-page PDF. It
reads the organization, title, and demand content cells, including
continuation rows. It admits only the 113 forms marked for full
disclosure. The 24 marked no, eight marked partial, and 61 with
unresolved selections are excluded. The Fujian parser checks 12
numbered tables with 109 published rows and joins split cells across
pages. Contact columns, credit codes, and names of contact people are
outside both projections.

Replay the complete [355-group regional frame](../enterprise-regional-frame-v3)
and all 4,301 historical records before adding the new sources.
Apply the same issuer grouping, exact text checks, four-gram overlap
tests and minimum length. In a separate admission screen, treat all
earlier public and historical records as exposure. Two hash seeds
reproduce the aggregate report and private frame byte for byte.

| Compilation | Listed forms | Complete with issuer and eligible disclosure | New candidate groups |
| --- | ---: | ---: | ---: |
| Yangzhou 2023 | 206 | 112 | 89 |
| Fujian 2025 | 109 | 107 | 86 |

Seven complete rows fall below the inherited length threshold. One
new row is connected to the historical corpus. The joint replay
retains all 355 prior group signatures and generates 175 further
candidate groups. These are proposed groups. Organization aliases,
civil-task eligibility, currency, and dependence across issuers
remain to be checked before sampling or locking a test set.
The frame has no retrieval outcomes or expert labels.

The open embedding screen on ws6 (Slurm job 47561) reproduced its
results in both orders. At cosine 0.85, one added group surfaced
against the historical corpus and none against the earlier frame
or the other added groups. No added group reached 0.90 in these
comparisons. The window projection and similarity scores are tools
for triage. A separate issuer-string comparison surfaces two focal
groups at 0.85 among the additions and zero at 0.90. The strongest
below-floor pairs form a 48-case packet across the three exposure
strata, shared by two blinded reviewer copies. Two parser and packet
replays reproduce the files byte for byte. The packet contains no
completed human judgments. It does not certify independence or
estimate the number of missed duplicates.

[Frame counts, pinned source digests, and semantic-screen counts](results.json)

The [eligibility pilot](../enterprise-eligibility-pilot-v1) freezes
65 groups for an independent original-source audit. Those groups
are reserved for the pilot and excluded from the later locked test.


## Reproduce

Use the four public snapshot groups and historical workbook referenced
by the previous [regional frame](../enterprise-regional-frame-v3#reproduce).
The PDF parser uses `pdfplumber==0.11.9`. A private input is required
for the historical overlap screen. Store private outputs separately.

```sh
umask 077
PYTHONPATH=src:. python -m experiments.enterprise_public_compendium \
  EARLIER_SNAPSHOTS MUNICIPAL_SNAPSHOTS REGIONAL_SNAPSHOTS PRIVATE_WORKBOOK \
  benchmarks/enterprise-sealed-first-query-v1/protocol.json \
  benchmarks/enterprise-regional-frame-v3/results.json PRIVATE_355_INDEX \
  benchmarks/enterprise-municipal-frame-v2/results.json \
  YANGZHOU_PDF FUJIAN_PDF NEW_AGGREGATE PRIVATE_530_INDEX \
  --projection PRIVATE_MASKED_VIEWS
```

The optional projection contains contact-masked text windows and
positional metadata for the semantic overlap replay. The issuer
grouping and source text remain local.

```sh
PYTHONPATH=src:. python -m experiments.external_enterprise_overlap gpu \
  PRIVATE_MASKED_VIEWS PINNED_OPEN_EMBEDDING NEW_PRIVATE_SCREEN --order-offset 0
```

The semantic replay runs inside a Slurm GPU allocation and repeats
with order offset 1.

For the review packet, rerun the compendium command with fresh
aggregate, frame, and projection paths, adding
`--overlap-review PRIVATE_SEMANTIC_SCREEN NEW_PRIVATE_REVIEW_DIR`.
The screen must match both the reconstructed frame and the text
projection hashes. The review copies omit channel, score, and
exposure status. Review intent and organizational linkage
independently; explicit uncertainty remains pending. Use the registered
checker after both reviewers return their copies.

```sh
PYTHONPATH=src:. python -m experiments.check_enterprise_panel_review \
  PRIVATE_REVIEW_DIR REVIEWER_A_JSONL REVIEWER_B_JSONL
```
