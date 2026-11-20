# Resolve original-source links for the CoF blind packets

Keep the [blinded depth-review packets](../cof-profile-depth-review-v1)
and the original CoF catalog snapshots fixed. Rejoin each
citation case to its exact frozen source ID in the custodian
map and confirm membership in the indexed reviewer history.
Use the original history to recover anchor source IDs. Match
only normalized titles when joining returned registry records
to the indexed title, and keep missing or discrepant records
unresolved.

Send only public work IDs to
[OpenAlex](https://help.openalex.org/api/) for KDD and
[Semantic Scholar](https://api.semanticscholar.org/api-docs/graph)
for SciRepEval. No query text, reviewer IDs or case codes
are sent to the registry endpoints. Fetch in batches and keep
the metadata caches local. The renderer shows sources and
separate profile anchors in blind form order. A lookup return
is not a full-text availability test. An open-access URL is
an access lead and may have changed.

A profile identity lead requires the intersection of bylines
from at least two unambiguous anchor works, deduplicated by
reported DOI where available, to consist of exactly one
registry author ID. The score also counts whether that same
ID occurs on the cited work's byline. These are identity
diagnostics, not judgments about individual contribution.
The numeric KDD reviewer IDs did not match the inferred
OpenAlex author ID in any of its seven resolved profiles.
Retain both namespaces.

For a paired locating comparison, use the completed
[Crossref title metadata lookup](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)
over the same citation cases. The title route accepts a
unique normalized exact title among its first five returned
results. An exact title does not establish an exact version.
Links found only by title, and DOI disagreements, are visibly
flagged for the reviewer rather than substituted into the
indexed record.

## Results

| Panel | Citation cases | Work-ID and title match | Work-ID DOI | Title-only route yields a DOI | Open-access URL supplied by registry | Unique author across anchors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KDD | 63 | 53 | 49 | 28 | 20 | 7 / 16 profiles |
| SciRepEval | 114 | 109 | 104 | 103 | 11 | 20 / 30 profiles |

Both routes yield a DOI in 125 citation cases, including
one disagreement. The title route contributes six leads
where the identifier route has no DOI. The inferred author
ID appears on the cited byline in 19 KDD and 70 SciRepEval
cases. No contribution label is inferred from authorship.
The summary records the registry and title-cache hashes
so offline rendering can be checked against the same
snapshots. [Aggregate results](results.json).

## Reproduce

Run with the private packet and the pinned CoF corpus.
Pass the local Crossref cache to include the paired
comparison. Use `--offline` to render from an existing
registry snapshot.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_review_source_ids \
  BLIND_PACKET COF_CORPUS_ROOT LOCAL_METADATA_CACHE NEW_REFERENCE_DIRECTORY \
  --offline --title-cache LOCAL_CROSSREF_JSONL
```
