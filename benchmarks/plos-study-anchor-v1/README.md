# Preserve ambiguous title anchors when a source is unavailable

This study precedes the
[report provenance check](../plos-study-provenance-v1).
Reconstruct the same three pinned PLOS publisher captures.
Extract normalized title tokens from 244 unique article records.
Use the first acquisition panel for threshold calibration and
the 163 titles first appearing in the later captures for testing.

Generate one complete title, a deterministic six-token contiguous
fragment, and the same fragment with one interior token deleted
for each article. For articles with a shared contiguous trigram,
also generate one such ambiguous clue. Replay each clue with the
target report available and with its record withheld from lookup.
The frozen-index route retains title identities even when their
contents cannot be retrieved. This simulates known unavailable
reports. Unknown records outside the acquisition manifest do
not produce such entries.

Compare title BM25 top one, per-view calibrated BM25, literal
matching over visible records, and literal matching over the
complete frozen title inventory followed by an availability
check. Calibrate the BM25 cutoff with the empirical 95th percentile
of unavailable-source scores in the first acquisition and require
a strict exceedance. A literal match looks up the least frequent
query trigram, then compares the full ordered clue against each
candidate title. A literal anchor confirms a source only if one
title matches in the relevant inventory.

## Results

Two CPU replays under different Python hash seeds agree on
all retrieval and error counts. Cells show correct available
results and false matches when the target is withheld.

| Clue | Cases in each visibility mode | BM25 calibrated | Visible literal | Frozen literal |
| --- | ---: | ---: | ---: | ---: |
| Complete title | 163 | 163 / 10 | 160 / 3 | 160 / 0 |
| Exact six-token fragment | 163 | 151 / 6 | 157 / 6 | 157 / 0 |
| Fragment with one token deleted | 163 | 134 / 6 | 0 / 0 | 0 / 0 |
| Shared trigram | 68 | 2 / 1 | 0 / 46 | 0 / 0 |

The frozen inventory prevents an unavailable match from turning a
formerly ambiguous clue into a lookup for another report. The
trigram index stores 3,139 distinct keys. An initial materialization
of every span of three or more tokens contained 25,552 distinct keys
and produced the same decisions. The count describes logical keys
for this corpus. It does not measure resident memory.

Keep the frozen literal route for known unambiguous identifiers and
send damaged or paraphrased clues to candidate generation. A lexical
score alone still makes wrong source assignments under withholding.
These authored title perturbations do not test extraction from
unstructured user requests, corpus updates, or end-to-end search.

[Aggregate results](results.json)

## Reproduce

Restore the three capture directories listed in the
[initials collision benchmark](../plos-study-bound-initials-v1).
The runner checks the XML hashes. The release contains aggregate
counts and digests.

```sh
PYTHONPATH=src:. python -m experiments.credit.study_anchor_lookup NEW_RESULTS \
  LEGACY_SNAPSHOT INDEPENDENT_SNAPSHOT BODY_TRANSFER_SNAPSHOT
```
