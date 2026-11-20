# Test title and author anchors against real bibliography targets

Use the 322 research articles in the pinned
[eLife XML manifest](../elife-role-nomination-v1/manifest.json).
Check each included article's XML hash. Read DOI, title, first author
surname, and electronic publication year. Read the corresponding
fields from bibliography entries in the same articles. Keep
non-self citations with a DOI, title of at least six tokens, first
surname, and four-digit year. For repeated cited DOIs, use the first
qualifying occurrence in filename and bibliography order. Citation
DOIs supply evaluation labels and are withheld from all retrieval arms.

Compare a unique title match alone, with the first surname, and
with the surname and year. Use the full text of the citation's title,
a deterministic three-token contiguous fragment, and a deterministic
six-token fragment with one interior token deleted. Only the latter
uses the [omission rule](../plos-study-omission-v1).
The shorter clues are constructed from real bibliography titles.
A separate controlled replay uses the uncorrupted author and year
from publisher metadata for each of the 317 eligible catalog articles,
then removes the intended target from the eligible matches.

## Results

The bibliography contains 19,766 distinct qualifying citation DOIs.
Six occur in the catalog. All 19,760 other DOIs are treated as absent.
This is exact DOI evaluation. Alternate versions and equivalent
publication records were not reconciled across DOIs.

| Clue | Title only wrong | Title and surname wrong | Title, surname and year wrong |
| --- | ---: | ---: | ---: |
| Full cited title | 3 | 1 | 0 |
| Three-token fragment | 363 | 5 | 1 |
| Fragment with one omission | 8 | 1 | 0 |

All arms recover five of the six in-catalog targets from the full
citation title and all six from each constructed fragment. The six
cases provide limited evidence about real-citation recall. In the
controlled publisher metadata replay, surname and year resolve
317/317 short fragments, versus 312/317 using title alone.
Removing the target produces four incorrect unique title matches
and zero after applying surname and year. The one-omission route
finds 317/317 with zero wrong outputs in that replay.

The remaining real bibliography error uses the same surname and
year on two related papers with different titles and DOIs. The
three-token clue matches both titles. Treating it as an identity
match would attach the wrong paper to an author's evidence. Bibliographic
metadata is cheap to check and substantially reduces false candidates
here; a DOI or independent provenance remains necessary for source
binding. The experiments measure one hop of source resolution.
They do not measure expert-level quality, service latency or
bookkeeping when citation metadata are missing.

[Aggregate results](results.json)

## Reproduce

Restore the XML included in the pinned eLife nomination manifest.
The experiment can run on a CPU and publishes only aggregate counts.

```sh
PYTHONPATH=src:. python -m experiments.credit.study_citation_anchor \
  benchmarks/elife-role-nomination-v1/manifest.json XML_DIRECTORY NEW_RESULTS
```
