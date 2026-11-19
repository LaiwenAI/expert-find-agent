# Combine source and role checks in open publisher-note search

Combine the frozen [source gate with full-text fallback](../plos-body-confirmation-v1) and [role applicability gates](../plos-role-applicability-v1) on the three-snapshot PLOS catalog. Use the same exact-study queries as the [public source-transfer benchmark](../plos-body-transfer-v1). The third snapshot's query DOIs and normalized byline names are disjoint from the two earlier snapshots. Ground truth comes from article identities and the publisher's contributor statements.

For each query, request authors with two duties in one particular study. Supply one legacy duty by name and the other through a hash-selected activity from the NISO guide. The matching activity is the one missing from a partial-match author in the controlled contrast. The alternative asks about an unsupported activity while preserving the source clue and the named duty. Independently vary source availability by keeping or removing the intended publication. Use both held abstract excerpts and keyword clues. One embedding determines each activity's role scores; another drives live source retrieval using dense and lexical ranking.

The following arms use the same source nomination and exact author intersection. No threshold is tuned on this panel.

| Policy | Source confirmation | Role selection and gate |
| --- | --- | --- |
| Forced | None | Enriched definitions, always select the highest-scoring legacy role |
| Relative | Dense score, then body fallback if needed | Legacy headings and top-two margin |
| Outside contrast | Dense score, then body fallback if needed | Legacy headings contrasted with excluded definitions |
| Definition contrast | Dense score, then body fallback if needed | Enriched definitions contrasted with excluded definitions |
| Definition contrast, no source gate | None | Same enriched role choice and gate |

The task requires the intended article and both duties to be directly credited to the same author. A recommendation citing another article is a source error even if that other article establishes related expertise. Report unsupported recommendations attributed to the wrong source separately from wrong roles on the intended source. Author matching uses exact publisher fields and filters ambiguous initials in the input parser; it does not estimate author identification from unstructured documents.

This is factored component timing. Each route is charged its live source search and role scoring, the source check when applicable, and binding. The role controller's shared per-request time includes all role scores. It excludes index preparation, HTTP transport and queues. The selected wording examples repeat across papers, so uncertainty across author groups is conditional on the tested wording. No inference about open industrial requests is made from the bounded scientific catalog.

## Results

Slurm job 42867 ran on one RTX 5090. The full catalog contains 197 publications. The 72 queries form 70 author-name components. The exact target was nominated for 70 excerpt queries and 57 keyword queries before source checks. The table lists supported requests. Withheld rows can only return a wrong source or abstain.

| Policy | Excerpts: clean / wrong role / wrong source | Keywords: clean / wrong role / wrong source | Withheld: wrong source, excerpts / keywords |
| --- | ---: | ---: | ---: |
| Forced | 56 / 14 / 1 | 47 / 10 / 11 | 56 / 48 |
| Relative + body | 4 / 0 / 0 | 3 / 0 / 0 | 0 / 0 |
| Outside contrast + body | 9 / 0 / 0 | 5 / 0 / 0 | 0 / 1 |
| Definition contrast + body | 18 / 13 / 0 | 14 / 9 / 1 | 0 / 2 |
| Definition contrast, no source gate | 19 / 14 / 1 | 17 / 10 / 6 | 29 / 24 |

For unsupported activities with the target article present, the forced route returns 66 / 63 recommendations (excerpts / keywords). The relative route returns one per clue format. Both outside-contrast routes abstain on all tested unsupported activities with or without the target paper. Relative and outside contrast use the same role ranker but separate frozen thresholds, so this compares complete operating policies rather than a score at matched recall.

With enhanced definitions fixed, source confirmation suppresses wrong-source recommendations when the intended paper is missing. Its cost is the loss of one excerpt and three keyword clean successes on the source-present queries. This confirmation does not fix confusing a supported role with a different supported role. For example, enhanced definitions with source confirmation improves excerpt clean completions from nine to 18 over the simpler outside-contrast route, but adds 13 wrong-role recommendations. Factored P95 is about 15.3 ms for excerpts and 14.5 ms for keywords. The richer route is a candidate generator, with room for a separate semantic verdict before attachment.

[Aggregate results](results.json)

## Reproduce

Use the three pinned captures from the [body-transfer benchmark](../plos-body-transfer-v1), the same guide JSON as the applicability gate, and the frozen result JSONs used as calibration inputs.

```sh
PYTHONPATH=src:. python -m experiments.credit.open_role_applicability \
  OLD_PUBLISHER_SNAPSHOT FIRST_TRANSFER_SNAPSHOT FRESH_TRANSFER_SNAPSHOT NISO_EXAMPLES \
  ROLE_APPLICABILITY_RESULTS OPEN_SOURCE_RESULTS BODY_CONFIRMATION_RESULTS \
  EMBEDDING_CHECKPOINT NEW_RESULTS.json
```
