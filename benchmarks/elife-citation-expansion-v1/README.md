# Expand the real-citation positives in the eLife source lookup test

The [original bibliography replay](../elife-citation-anchor-v1)
contains only six in-catalog citations. Among citations whose DOI
is missing from that catalog and matches the eLife five-digit article
identifier format, sort by SHA-256 of the fixed experiment name plus
the citation DOI. Fetch the first 192 version-one XML files from the
same pinned publisher revision. Verify exact XML hashes using
[the download receipts](receipts.json). Accept only research articles
with the exact cited DOI and complete title, first surname, and
unambiguous electronic publication year. Recognize both electronic
publication date layouts used in the publisher snapshot.

All 192 selected reports pass. Keep the original 322 articles as
distractors. Replay the 19,760 genuinely external bibliography targets
from the prior study against the enlarged 514-article directory. Use
the same rules and clue construction. DOI is used for sampling and
evaluation; retrieval receives the title clue and optional bibliography
surname and year. No parameters are tuned on the new positives.

## Results

Each cell reports correct reports out of 192 available targets and
wrong reports among 19,568 targets remaining outside the directory.
Unreturned cases are unresolved. None of the arms had any wrong
outputs among the available targets.

| Clue | Title alone | Title and first surname | Title, first surname and year |
| --- | ---: | ---: | ---: |
| Full real citation title | 188 / 3 | 185 / 1 | 185 / 0 |
| Three-token title fragment | 182 / 440 | 187 / 5 | 187 / 1 |
| Six-token fragment with one interior omission | 188 / 7 | 187 / 1 | 187 / 0 |

Among the new positives, cited and catalog years agree for 192/192.
Normalized first surnames agree for 189/192. The three discrepancies
are partial compound surname, diacritic variation, and a different
surname in the bibliography. Consequently exact metadata matching can
disambiguate short clues while losing some long-title correct hits.
Keep conflicts eligible for a separate provenance check, especially
when the full title uniquely points to a candidate. The earlier
wrong match between two related same-author papers remains possible
for the short fragment. Source binding requires stronger evidence.

The same-publisher, DOI-sampled positive panel measures source identity
under genuine bibliography variations. The short and omission clues
are controlled transformations. The exercise evaluates neither the
prevalence of such clues in enterprise requests nor request latency.
Two CPU runs under different hash seeds produced identical outputs.

[Aggregate results](results.json)

## Reproduce

Restore the original XML from the
[eLife nomination manifest](../elife-role-nomination-v1/manifest.json).
Download the filenames named in the receipts into a separate directory
from the publisher path
`https://raw.githubusercontent.com/elifesciences/elife-article-xml/58250dd64e27afae6bc2700bcddd84dd38776e0d/articles/<filename>`.

```sh
PYTHONPATH=src:. python -m experiments.credit.study_citation_expansion \
  benchmarks/elife-role-nomination-v1/manifest.json ORIGINAL_XML \
  run ADDED_XML benchmarks/elife-citation-expansion-v1/receipts.json NEW_RESULTS
```
