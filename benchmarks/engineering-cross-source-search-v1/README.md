# Test role normalization and same-owner proof in an engineering search pool

Add six open-access Sensors publications to the
[dual-format engineering collection](../engineering-contribution-transfer-v1).
Each new publication has a publisher ORCID matching one of the original
contributors. Verify each record's DOI, XML checksum and license, then
check the link in the source and destination by checksum-valid ORCID and
author ordinal. Unlinked authors remain local to their publication. Both
publisher role representations must agree. There are 60 licensed articles
before abstract filtering and 53 indexed articles afterward. All six
verified links remain; four pairs produce eligible role-swap controls.

Each of 15 families supplies two source descriptions and one role per
source. On the positive example at least one author holds both roles at
their designated sources. In the paired control swap the role-to-source
binding. Each role must appear somewhere on the opposite article, and
no author may satisfy the resulting declared conjunction. Choose at
most five families per pair by deterministic order, favoring role
complementarity. Neither the author nor an article identifier is in
the request. The benchmark provides the two clause-to-source slots.
It does not attempt unrestricted parsing of those slots from prose.

## Retrieval and resolution

Index the title and first half of each abstract. Query using the
remaining abstract sentences, or up to eight of their high-IDF keywords.
Encode both source descriptions and both synthetic role requests in
one Qwen3-Embedding-0.6B batch. Each role uses a fixed example authored
by [NISO](https://zenodo.org/records/18421449) for that family. The
catalog combines dense and BM25 ranks by reciprocal-rank fusion.
Search the same 53 articles for every query. The missing-source controls
remove one of the two target publications.

- **Dense or hybrid first:** use the first result on each side and
  check both role requirements by the same owner key.
- **Linked:** among the top six results on each side, pick the highest
  rank-sum pair with a verified author link, then check the roles.
- **Supported:** pick the highest rank-sum pair with a declared witness
  for every required role for one contributor. A missing target source
  can make this select a wrong but supported substitute.
- **Anchored:** use the support rule and keep at least one original
  first-ranked hybrid source. Both source scores must also be within
  0.04 cosine of their independent dense leader. This is a fixed
  heuristic, without fitting a source threshold.
- **Oracle sources:** supply the two designated publications. If one
  is missing, return no answer. This isolates role normalization.

Compare oracle roles with the
[frozen ridge and prototype role banks](../engineering-role-banks-v1).
Top-one policies use the best role on each side. The top-one agreement
route requires both banks' best roles when they differ. The band routes
require a candidate to hold every role within the independently frozen
single-role margin of the best role. Both margins use the Frontiers
calibration for transfer to Sensors. No thresholds are fitted to this
linked panel.

## Results

Both Slurm tasks of array 45897 on ws3 RTX 5090 reproduce every
non-timing field. All 30 requests have both designated documents
in their fused top-six lists with either clue representation.
With oracle roles and keyword clues, the following table counts
correct sources across both variants and exact positive proof
across the 15 positives. The last column includes both variants
and both missing-source removals.

| Policy | Exact source pairs / 30 | Exact positives / 15 | Substituted outputs / 60 with a missing source |
| --- | ---: | ---: | ---: |
| Dense first | 14 | 7 | 0 |
| Hybrid first | 20 | 10 | 0 |
| Linked | 30 | 15 | 5 |
| Supported | 15 | 15 | 5 |
| Anchored | 10 | 10 | 0 |

With excerpt clues and oracle roles, independent hybrid retrieval
already recovers every designated pair. The supported route still
substitutes other linked papers on seven swap cases. Searching for
a pair that satisfies the requested roles can invalidate a role
rejection on correctly retrieved papers.

With oracle sources, use both role clauses to audit the language
gate. Correct authors can coincide even when the inferred roles
are wrong. The proof column also requires both intended roles
to be among the declared roles explicitly checked.

| Role mode | Exact author sets on positives / 15 | Exact positives with both intended roles / 15 | Unsupported outputs on swapped roles / 15 |
| --- | ---: | ---: | ---: |
| Oracle roles | 15 | 15 | 0 |
| Ridge top-one | 8 | 3 | 2 |
| Ridge band | 1 | 1 | 0 |
| Prototype top-one | 7 | 4 | 2 |
| Prototype band | 4 | 4 | 0 |
| Top-one agreement | 7 | 4 | 2 |

The same-owner source link improves recoverability in the keyword
control, while widening the opportunity for source substitution.
Single-role calibration transfers poorly to the two-role proof even
when the correct sources are supplied. The calibrated ridge band
includes both intended roles on 28 of 30 cases, but its extra roles
usually leave no fully supported contributor. These are four
engineering source pairs with repeated labels and wording. They do
not estimate open-population accuracy or substantiate expert
competence beyond the source declarations.

Warm factored P95 is approximately 10.1 ms for keyword clues and
12.6–12.8 ms for excerpts. Count one live embedding batch and topic
ranking for each policy, even when shared for the comparisons.
Time the resident join and serialization. Indexing, downloading,
service transport, and raw user-request parsing are outside the timer.
[Metrics](results.json) · [Compressed source-selection trace](live-evidence.json.gz).

## Reproduce

First restore the
[engineering publisher panel](../engineering-contribution-transfer-v1).
Preserve its original XML and write its audit as
`SEED_SNAPSHOT/audit-confirmed.json`. Restore the six new projections
using the [link intake manifest](linked-intake.json).

```sh
PYTHONPATH=src:. python -m experiments.credit.engineering_cross_source_panel \
  SEED_SNAPSHOT NEW_LINKED_SNAPSHOT PANEL NEW_MANIFEST \
  --restore-linked benchmarks/engineering-cross-source-search-v1/linked-intake.json

# Inside a Slurm CUDA allocation. Repeat with offset 1.
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.engineering_cross_source_search \
  PANEL NEW_MANIFEST NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-role-banks-v1/results.json QWEN_EMBEDDING NEW_RESULTS
```

The source restore reproduces the derived panel byte for byte.
The manifests and results contain no bylines or source passages.
