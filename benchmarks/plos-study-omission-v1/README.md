# Recover report title clues containing one missing interior token

Use the pinned 244-title PLOS catalog and the chronological split from
[the literal-anchor study](../plos-study-anchor-v1). The first 81 records
calibrate BM25. The remaining 163 reports each supply two controlled
five-token clues. The first deletes one interior token from a contiguous
six-token title span. The second also substitutes one of the remaining
interior tokens with a frequent catalog token absent from the clue.

An interior deletion leaves the leading or trailing trigram of the
clue intact. Look up both trigrams in the existing index, verify the
six-token spans of the retrieved titles, and union the verified matches
with all exact five-token matches. A match is unique only if that
union contains one report. Check source availability after checking
uniqueness. The visible-only variant checks availability first.

## Results

Each mode contains 163 cases of each clue type. The table lists correct
and wrong outputs when the target is available and wrong outputs
under each outage scenario. Any other case is unresolved.

| Clue | Route | Available correct / wrong | Wrong with registered target unavailable | Wrong after removing target from matching inventory |
| --- | --- | ---: | ---: | ---: |
| One omission | Frozen literal | 0 / 0 | 0 | 0 |
| One omission | Calibrated BM25 | 134 / 3 | 6 | 6 |
| One omission | Visible-only omission | 157 / 0 | 6 | 6 |
| One omission | Frozen omission | 157 / 0 | 0 | 6 |
| Omission and substitution | Frozen omission | 3 / 0 | 0 | 0 |
| Omission and substitution | Calibrated BM25 | 113 / 4 | 7 | 7 |

Removing a target from the matching inventory omits its postings
from eligibility while retaining the frozen calibration and BM25
collection statistics. It isolates the effect of unknown catalog
membership. Outage rows have no correct report by construction.

Across the 163 one-omission clues, the two trigram lookups shortlisted
186 titles in total, counting repeat lookups across queries. The
existing 3,139 trigram keys suffice. Materializing all omission
signatures would add 10,020 keys and 10,160 report postings.
An independent enumeration checked equivalence over all 12,763
distinct eligible exact or omitted spans. Two CPU runs with different
Python hash seeds produced identical aggregate JSON. These counts
measure lookup work on this catalog.

The known-outage result depends on retaining the target's catalog entry.
Six ambiguous clues become incorrectly unique after that entry disappears.
The route also offered limited recovery for a second corruption. Use it
to nominate source candidates. Require independent source provenance
before using a recovered report as evidence for personal contribution.
Natural clue extraction, multilingual variation and complete request
latency remain to be evaluated.

[Aggregate results](results.json)

## Reproduce

Restore the same captures as in the
[literal-anchor study](../plos-study-anchor-v1). The loader checks XML
hashes before indexing.

```sh
PYTHONPATH=src:. python -m experiments.credit.study_anchor_omission NEW_RESULTS \
  LEGACY_SNAPSHOT INDEPENDENT_SNAPSHOT BODY_TRANSFER_SNAPSHOT
```
