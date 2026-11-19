# Measure provenance availability for CoF citation review

Audit the frozen citation-review packet from [the full-inventory CoF replay](../cof-citation-package-review-v1/README.md). The packet contains cited titles and three separate profile-title anchors per candidate. It supplies no official identity crosswalk or author-contribution labels.

Search each distinct public title through the [Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/), requesting the first five results. Match normalized titles exactly and require a unique DOI among the returned matches. Distinguish unmatched titles, multiple exact DOI candidates and request failures. Normalize full byline names and retain checksum-valid ORCID identifiers when the deposit provides them. For each profile require at least two distinct exact anchor publications with usable bylines. Intersect all usable bylines. A single common name is a candidate identity, with stronger support if the same valid ORCID is present in at least two anchor records. Conflicting ORCIDs remain unresolved. This heuristic does not establish that the index's candidate identifier maps to a person with certainty.

Check whether the candidate's name or confirmed ORCID appears on the cited byline. For exact citation DOIs, use the [Europe PMC REST service](https://dev.europepmc.org/RestfulWebService) to check for open XML. Inspect that XML for a contribution section, a contribution note or structured author roles. Separately count a structured role that can be linked to the candidate by name or ORCID. The audit measures coverage of these public metadata surfaces. It does not assess whether a declared role supports a specific query, and other publishers or licensed files may offer evidence beyond these surfaces.

Only public paper titles go to Crossref and only resolved public DOIs go to Europe PMC. Raw title-to-author records stay in the local cache. Requests are sequential and paced for the public API. The aggregate contains counts without names, identifiers, abstracts or role text.

## Verified audit

The lookup covered all 232 distinct paper titles. Crossref returned one normalized exact DOI for 163 titles, multiple exact DOIs for 17, and no exact title among the first five results for 52.

| Panel | Citation cases | Unique exact citation DOI | Candidate slots | One common byline name across anchors | Repeated valid ORCID across anchors | Common candidate name on cited byline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KDD | 82 | 47 | 32 | 14 | 0 | 21 |
| SciRepEval | 36 | 31 | 17 | 5 | 0 | 7 |

The citation matches represent 63 distinct DOIs. Europe PMC completed lookups for 39 and gave no unique open XML match in those responses. Lookups for 24 DOIs failed after retries. These requests account for 25 citation cases and remain unresolved. The audit retrieved zero XML records and has no evidence about the prevalence of personal contribution statements.

The common-name links are leads for review. Twenty-one KDD and seven SciRepEval cases also contain the proposed name on the cited byline, which establishes only a bibliographic match under this heuristic. Decisive author-level evaluation needs a source that links a particular person to specific work. This blind packet cannot be used as ground truth for personal attribution based on registry coverage alone. [Aggregate results](results.json)

## Reproduce

Use the frozen local CoF review packet. The Crossref title search is live, so exact outputs can change with metadata or service coverage. A local cache permits a paused capture to resume.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_public_provenance_audit \
  LOCAL_COF_REVIEW_PACKET LOCAL_CACHE NEW_RESULTS
```

## Local reference sheets

Generate arm-blind Markdown sheets from any completed portion of the cache. The renderer verifies the frozen review forms, ignores an incomplete final cache line, and includes only exact DOI records. It retains the citation and profile titles in their fixed order and does not copy raw registry bylines, identity hypotheses, ranks, grades or method identities. The original review forms remain immutable. Keep the sheets local with the underlying packet.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_review_references \
  LOCAL_COF_REVIEW_PACKET LOCAL_CACHE NEW_REFERENCE_DIRECTORY
```
