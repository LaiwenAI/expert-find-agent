# Registry-link reach in the enterprise first-query panel

The [structured patent replay](../patent-inventor-replay-v1/SERVING.md)
started with inventor rosters from PatentsView. This audit measures
whether the first-query enterprise search has corresponding lookup
anchors or named inventor fields in its current profile inventory.

Reuse the frozen inventory and
[locator screen](../enterprise-source-linkability-v1), and preserve
the 80 development requests, the first five candidates, their planned
excerpts, and the hybrid-prefix arm's output. Scan two granularities.
The original profile field may list multiple works. Each delimited
clause is a separate candidate record. Count a clause as addressable
when the existing locator scanner yields one distinct normalized
typed string in that clause. Keep literal namespaces and retain the
kind-code or check-character spelling when present. No
external registry lookup or checksum validation is performed. No
private identifiers were sent to a registry.

A separate syntax stratum detects modern Chinese application-number
shapes with the 12-digit year, type, and sequence structure described
by [CNIPA](https://www.cnipa.gov.cn/art/2003/7/14/art_526_145959.html).
Distinct strings may denote the same work. A syntactic key establishes
a lookup lead and does not establish a valid filing, exact source
edition, inventorship, or evidence for a requirement.

For same-identifier patent clauses across different profiles, require
one unconflicted inventor list in a clause carrying the identifier.
Count a possible supplement only when the exact normalized candidate
name occurs in that list, the name is unique within the fixed catalog,
and the receiving clause lacks a local list for that identifier.
This is a profile-attribution cue without separate identity verification.

## Results

| Patent locator reach | All development requests | Requests with initial abstention |
|---|---:|---:|
| Any locator-shaped span in the top five profiles | 70/80 | 47/55 |
| Exactly one typed string at whole-field level in at least one profile | 14/80 | 8/55 |
| At least one typed clause in at least one profile | 66/80 | 45/55 |
| At least two typed clause strings under the same candidate | 65/80 | 44/55 |
| At least one modern CN application-number shape | 64/80 | 44/55 |
| Exact owner name in a local inventor clause | 1/80 | 1/55 |

The 455 patent fields with typed clauses contained 3,067 clause
occurrences. The most common form was the modern CN application
shape, with 2,225 distinct owner–string pairs. Of 799 planned
excerpts, 77 fully contained at least one typed patent clause,
distributed across 49 requests. The 31 patent identifier groups
shared across profiles contained zero parseable inventor lists in
the identifier-bearing clauses, yielding no new named cue through
local pooling.

The highest-coverage next source is a licensed or authorized CN
bibliographic snapshot joined in the restricted workspace. This audit
measured only formatting and reach. It did not estimate how many
records would resolve or how many unreturned candidates actually
qualify. The existing full-text path retains candidates with missing
registry evidence.

## Reproduce

The preparation step binds the existing restricted source-cue
projection to the manifest and writes the smallest projection used here.
Keep that projection, the raw catalog, the frozen trace, and review input
outside the public repository. The audit computes only integer aggregates.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_registry_reach prepare \
  PRIVATE_DATASET PRIVATE_CUES PRIVATE_PROJECTION
python -m experiments.retrieval.enterprise_registry_reach audit \
  PRIVATE_PROJECTION FROZEN_DEVELOPMENT_TRACE FROZEN_CORPUS RESULTS
```

The aggregate was reproduced byte-for-byte on ws3.
