# Conservative role postings for first-query expert ranking

Build a resident index of publisher-declared roles. For each requested role, a candidate is possible if an indexed declaration confirms it or the candidate has an opaque declaration. Intersect these candidate sets across roles. Materialize the canonical witnesses in the original source order, falling back to an opaque declaration if it could provide the earliest citation. A separate forward-cache control uses the same indexed declarations and opacity flags without candidate postings.

A candidate removed by a posting has no positive or opaque witness for at least one criterion. It cannot produce a confirmed recommendation. Every candidate retained is visited in the common byline-based prior order and evaluated against its own records. The automatic route scans the forward cache if the intersection retains more than half the candidate pool. This threshold is a fixed heuristic. The index assumes ownership, source visibility, and version have already been checked. Missing or stale declarations must arrive as opaque.

## Data

The [fixture](sources.json) contains 152 locally available, hash-matched records from the prior 165-article [PLOS catalog](../plos-credit-external-search/README.md). Each record retains publisher DOI, XML hash and field, author ordinal, and role declarations. Validated publisher identifiers use dataset-local tokens. Authors without a validated identifier remain local to their article. Of 580 candidate identities, 564 have one publication, 14 have two, and two have three. Role declarations are credited to PLOS records under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Enumerate the 364 unordered triples of the 14 CRediT roles. Each request searches the entire visible snapshot and returns up to five ranked identities with one canonical source per role. Five nested coverage settings either mask individual declarations or mask whole articles. For every setting the 20 rows without declared roles stay opaque. Masked labels are available to the timed fallback callback, so masking models the contents of the resident cache rather than missing ground truth.

## Results

Every route matches the exhaustive ranked identities and all citations for all 364 queries at every setting. The fixed bundled route does not use the compiled cache. Warm CPU P95 is the quantile of seven per-request timings summarized by their median. Construction, XML reading, query parsing, semantic retrieval, inference and storage I/O are outside query timing. A fallback read is an in-memory lookup in this replay.

| Whole-article mask setting | Declarations indexed / 598 | Bundled P95 | Forward-cache P95 | Posting-only P95 | Auto P95 | Auto fallback reads per query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0 | 242.52 us | 22.40 us | 21.17 us | 21.05 us | 55.56 |
| 25% | 160 | 248.83 us | 20.61 us | 22.28 us | 19.15 us | 38.83 |
| 50% | 289 | 243.52 us | 17.43 us | 15.50 us | 16.40 us | 26.69 |
| 75% | 459 | 245.91 us | 15.69 us | 9.77 us | 9.42 us | 14.59 |
| 100% | 578 | 247.35 us | 12.92 us | 4.50 us | 4.33 us | 5.55 |

The same coverage pattern appears when masking declarations individually. At low coverage posting traversal can be slower than the forward cache. The cache accounts for all fallback-read savings: the forward and posting routes read exactly as many opaque declarations. At higher coverage the posting intersection reduces candidate visits and CPU time. These measurements cover a structured, publisher-declared task with a fixed prior and at most three publications per identity. The experiment does not measure how often free-text enterprise capabilities can be certified or claim a production latency gain. [Aggregate results](results.json)

## Reproduce

```sh
PYTHONPATH=src:. python -m experiments.credit.role_evidence_postings run \
  benchmarks/plos-credit-role-postings-v1/sources.json /path/to/new-results.json
```

To rebuild the fixture, run the same module with `prepare PINNED_CATALOG NEW_FIXTURE XML_DIRECTORY [XML_DIRECTORY...]`. The prior PLOS catalog builder is linked from the data section above. The result records the input, index and replay digests.
