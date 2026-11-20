# Paper locators and author clues in the enterprise first-query shortlist

Use the same frozen 80 development requests, top-five candidates and
evidence excerpts as the [first-round replay](../enterprise-joint-first-round-v1).
The linked snapshot contains 4,154 catalog profiles and 33,320 paper
rows. The audit does not send titles, names, locators or requests to an
external service. It leaves the original search and verifier unchanged.

Measure three kinds of available information:

1. **Location.** Report a DOI-shaped value, a numeric Scholar cluster
   value, and a syntactically valid HTTP(S) PDF pointer separately. These
   are saved metadata. Reachability, record correctness, access rights,
   and current versions have not been checked.
2. **Identity clues.** Screen each parsed author list for the profile's
   exact normalized name and a catalog-unique tone-free transliteration.
   Transliterate Han names in both family-first and given-first order.
   A collaborator clue requires one shared author token with a paper
   under that candidate's profile in which the original profile name
   appears exactly and is unique in the fixed profile catalog.
3. **Contributor evidence.** Count explicit verified roles carried by
   the input sidecar. The raw roster and this sidecar do not encode
   contribution roles. A zero count means the supplied paper projection
   offers no direct role evidence. Authors' contribution to the real
   work is unjudged.

For references from another profile, the title must differ from all
normalized titles already assigned to the candidate and no supplied
DOI or Scholar cluster value may overlap. Reject conflicting IDs when
counting clean locator groups. The distinct-title test is conservative
about exact duplicates. It does not resolve variants or independent
publications with identical titles. Author field parsing is a simple
delimiter screen. Name and collaborator hits are leads within this
catalog. They do not establish identity, eligibility or publication
relevance.

## Results

A request counts once when at least one of its five frozen candidates
satisfies a condition.

| Availability in existing material | All 80 requests | 55 with first-round abstention |
| --- | ---: | ---: |
| Candidate has a saved paper author list | 80 | 55 |
| Candidate has a saved paper DOI-shaped value | 69 | 49 |
| Candidate has a saved paper PDF pointer | 80 | 55 |
| Reviewed excerpt already selects a paper with DOI-shaped value | 17 | 11 |
| Reviewed excerpt already selects a paper with PDF pointer | 39 | 30 |
| Candidate has two distinct titles with exact catalog-unique name and clean saved ID | 37 | 26 |
| Another profile nominates a distinct title from a name match | 70 | 49 |
| That nomination also has a shared author and a clean saved ID | 6 | 4 |
| A candidate has a verified contributor role in the paper sidecar | 0 | 0 |

In the development inventory, 32,938 of 33,320 paper records have a
parseable author field. There are 5,257 DOI-shaped values and 17,386
PDF pointers. There are 208 DOI or Scholar keys associated with multiple
exact normalized titles. That can include ordinary metadata variants and
is sufficient to flag an uncertain join under this rule.

In the abstention cohort, 30 requests already have a selected paper
with a PDF pointer. That is the higher-coverage route for checking
original declarations under this inventory. The study establishes no
PDF retrieval coverage or contribution-quality gain. Cross-profile
name expansion has broad nomination reach. Its additional shared-author
criterion leaves only six candidate slots across four abstaining
requests. This yields a small targeted verification stratum. The
separate [patent audit](../enterprise-registry-reach-v1/CN-OFFLINE.md)
covers publication and inventor identifiers and has a different evidence
scope.

## Reproduce

Use the existing private development snapshot, metadata sidecar, raw
paper inventory and frozen search trace in the authorized workspace.
The script checks the declared file hashes and original record joins.
It writes only aggregate counts. Two local CPU replays under Python
3.11 and pypinyin 0.55.0 produced byte-identical results.

```sh
python -m experiments.retrieval.enterprise_paper_identity_audit \
  PRIVATE_DATASET PRIVATE_METADATA PRIVATE_RAW_PAPERS PRIVATE_TRACE NEW_AGGREGATES
```
