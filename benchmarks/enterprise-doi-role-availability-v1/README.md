# Audit access to personal contribution evidence from enterprise citations

Use the frozen private packet from the
[enterprise routed-review run](../enterprise-routed-review-v1).
Validate the exact exhibited source views and join the paper sidecar by
source ID and owner. Evaluate two DOI inventories. The direct inventory
contains the catalog DOI for each selected paper. The owner inventory
contains all stored candidate-owned paper DOIs. This larger inventory
measures the availability of alternative role material. It does not
measure whether an alternative is relevant to the original requirement.

Require exact requested-DOI and returned-DOI correspondence from the
[PMC ID Converter](https://pmc.ncbi.nlm.nih.gov/tools/id-converter-api/).
Fetch mapped articles through
[Europe PMC](https://dev.europepmc.org/RestfulWebService)
and verify the DOI within the article XML. Use a known open article
with a contribution declaration to exercise the entire lookup.
Record CRediT author-role elements and prose contribution markers as
separate outcomes. Match the candidate by an exact normalized full
name or by a script alternate explicitly supplied by the publication.
Also report unique phonetic name candidates separately. A transliteration
match without independent identity evidence is a review lead.

## Stored identifier result

| Scope | Cases with a stored DOI | Distinct DOIs | Mapped to PMC | Retrieved JATS |
| --- | ---: | ---: | ---: | ---: |
| Displayed papers | 3 / 71 | 3 | 0 | 0 |
| All records associated with the same candidates | 16 / 71 | 62 | 0 | 0 |

The packet contains 21 displayed paper citations. The positive control
mapped and its JATS contained contributor-role evidence. The archive's
lack of a mapping for a DOI provides no judgment about the paper's
availability elsewhere or an author's role. The fetch time for the
batch is a public service observation and is not part of the expert
search latency measurement. This audit uses lawful public metadata and
open archive access. Only aggregate counts and reproducible code are
released.

## Recover bibliographic identifiers from title and year

For each of the 21 displayed papers, verify the exact raw catalog row
and checksum, and send only the public article title to the
[Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/).
Inspect eight search candidates. Require an exact normalized title
and matching publication year when the local year is available. Retain
a DOI only when exactly one passes. A bibliographic match locates an
article, while a generic author role in Crossref's bibliographic metadata
provides no credit for an individual contribution.

All 21 requests returned metadata responses. Fourteen papers matched
a unique DOI with exact title and year. Eleven previously lacked a DOI
and three matched a DOI different from the value in the catalog.
Two of the 14 matched PMC identifiers. One returned JATS and lacked
the checked CRediT or author-contribution markers. The other XML
request failed in transport after a retry, so its contribution status
is unknown. This path produced no confirmed person-specific contribution
evidence in the 21-paper panel.

The distinct failure points are catalog linkage, open-archive access,
author identity, and explicit duty. Use them as separate outcomes and
evaluate publisher or alternative project sources under a bounded
lookup policy. The 21 papers come from the development recommendations.
Results do not estimate archive coverage over all catalog content.

[Aggregate results](results.json)

## Reproduce

Use the private development review packet and the original paper
metadata sidecar. Install `pypinyin==0.55.0` in the experiment
environment for the diagnostic phonetic-name comparison.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_doi_role_availability \
  PRIVATE_REVIEW_PACKET PRIVATE_PAPER_SIDECAR NEW_PRIVATE_CAPTURE
```

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_paper_identifier_repair \
  PRIVATE_REVIEW_PACKET PRIVATE_PAPER_SIDECAR PRIVATE_RAW_CATALOG NEW_PRIVATE_CAPTURE
```
