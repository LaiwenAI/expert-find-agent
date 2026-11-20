# Transfer joint contributor nomination to eLife

Rebuild the official-license 322-article [eLife projection](../elife-role-nomination-v1),
whose included source records exclude exact title, normalized author-name
and valid ORCID overlaps with 479 prior PLOS XML records. The restored
projection must match its published SHA-256. Role text attached to
individual contributors is parsed from publisher XML. A single
checksum-valid ORCID links a contributor across publications.

Transfer the [PLOS joint nomination protocol](../plos-joint-byline-depth-v1)
without tuning. Build two panels separately from publisher subject
headings and author keywords. A topic must appear on at least four
catalog articles. A request specifies two different topics and two
different roles, without naming a person or source. Each candidate who
can satisfy both conditions according to the reference must require
two different articles. The topic and contributor labels generate
and score the requests, outside retrieval. Cap the number of queries
per organizing contributor and deduplicate using the same PLOS rule.
An empty panel means this catalog could not supply qualifying queries
under the frozen construction.

Run the frozen BM25 and Qwen3 hybrid rankers and the same source
selection at depths 8, 16, 32 and 64. Each policy may read up to
three different publisher declarations. Record the available reference
evidence in the pools, the one-more-read ceiling after fixing the
first two sources and the results of topical, staged and joint
selection. Tag-unresolved returns include citations with an unlisted
topic label. The evaluation reuses a structured role taxonomy, source
identities and an official publisher vocabulary. It does not
establish expert suitability or service latency.

## Results

Both tasks in Slurm array `44438` ran on ws5 with an RTX 5090
and produced identical aggregates under distinct Python hash seeds.
The publisher-heading panel contains 51 requests over eight topic
pairs. The author-keyword panel contains 12 requests built from
one topic pair. Every hybrid policy in the table other than the
initial two-read reference issues exactly three declaration reads.

| Panel | Depth | Full proof reachable / requests | Topic-first known | Owner-gap known | Joint byline known | Joint byline and article-role union known |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Headings | 8 | 12 / 51 | 0 | 0 | 12 | 12 |
| Headings | 16 | 14 / 51 | 0 | 0 | 12 | 14 |
| Headings | 32 | 15 / 51 | 0 | 0 | 12 | 15 |
| Headings | 64 | 29 / 51 | 0 | 0 | 26 | 29 |
| Keywords | 8–64 | 12 / 12 | 0 | 0 | 12 | 12 |

Fixing the common first two articles leaves zero eLife requests
reachable with one extra article, at every tested hybrid depth.
The joint plan alters initial exposure and recovers known matches
under the same declaration read budget. At depth 64 the article
role union adds three heading matches to the byline-only plan.
Neither plan issues a tag-unresolved return in these hybrid
panels. That absence is descriptive. Eight heading clusters and a
single keyword pair leave limited evidence about other topic mixes.
This study transfers the source selection mechanism to another
publisher while keeping structured semantics and known ORCID links.
It does not estimate how missing or ambiguous identity metadata
would reduce the gains.

[Complete lexical and hybrid aggregates](results.json)

## Reproduce

Restore included eLife XML files using the pinned
[manifest](../elife-role-nomination-v1/manifest.json).
Run in a Slurm allocation with verified Qwen3 embeddings.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.elife_cross_publication_transfer \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING NEW_RESULTS
```
