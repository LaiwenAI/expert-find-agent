# Calibrate resident source scores on same-author pair proposals

The [resident source study](../elife-resident-source-v1) ranked all
five intended keyword pairs above its three wrong proposals, but the
generic document-level threshold rejected all pairs. Keep both fitted
PLOS feature models fixed. Recalibrate only their pair thresholds on
the archived [earlier eLife panel](../elife-prompt-holdout-v1) and
[initial eLife catalog](../elife-role-nomination-v1).
Check their manifest and projection hashes. Reject DOI and recorded-author
overlap with each evaluation panel.

The earlier corpus has 393 eligible documents before adding the
supplement. Keep one positive and one role-swap request per linked
paper pair in each snapshot and use their combined index. For each query,
consider top-24 source lists with both sources present and with each
target withheld. Take the four highest-ranked same-author pairs and
the four highest-ranked pairs compatible with dense singleton roles,
then remove any exact target. Add the known target when both
documents are in the top 24. Deduplicate by target, availability,
and proposed pair per view.

Score a pair by its weaker resident source score, and set the new
threshold to the largest wrong-pair score within each view. A view
without a true pair, wrong pair, and missing-target pair retains its
generic threshold. Neither
evaluation panel chooses a threshold. Reuse the same one-forward
role check, original cascade and dense fallback. Separate the
withheld-source test, which measures additional fallback outputs.
Repeat both request orders. This is a matched calibration diagnostic
on two previously examined evaluation panels, without service timing.

## Results

All tasks in Slurm array 46330 complete on ws3 RTX 5090.
Both request orders give identical outputs. The initial single-catalog
attempt in array 46323 stopped because calibration lacked a stratum.
The combined calibration uses 673 articles and 23 linked paper pairs,
giving 92 requests before candidate construction.

| Calibration view | Correct source pairs | Wrong pairs | Of wrong pairs, missing a target | Correct admitted, matched gold / distilled |
| --- | ---: | ---: | ---: | ---: |
| Keywords | 21 | 3 | 2 | 20 / 21 |
| Excerpts | 23 | 13 | 6 | 22 / 23 |

The wrong pairs are blocked by construction when choosing the cutoff.
The same calibrated thresholds are used for both evaluation panels.

| Policy | Development keyword strict / adverse | Held keyword strict / adverse | Extra role forwards, development / held |
| --- | ---: | ---: | ---: |
| Original cascade | 12 / 0 | 10 / 0 | 0 / 0 |
| Dense source-only fallback | 15 / 0 | 11 / 1 | 0 / 0 |
| Generic resident source gates | 12 / 0 | 10 / 0 | 0 / 0 |
| Matched gate plus 4B role check | 14 / 0 | 11 / 0 | 3 / 2 |

Both matched heads admit the three correct development source pairs
and both correct held pairs. The 4B role check rejects the held
swapped-role control and one development positive. Every excerpt
response is unchanged. The matched policy uses no extra source
forward. Each triggered request still uses at most one model forward.

With a target removed, development proposes eight new pairs and
the matched gate rejects all of them. The held panel produces no
such source proposal. The observed quality gain is conditional on
the frozen structured panels and their known keyword or excerpt view.
The keyword calibration includes only three wrong pairs, and the
held gain comes from a single query. A larger independent risk panel
and measured end-to-end latency are required. See
[aggregate results](results.json).

## Reproduce

Restore the calibration snapshot from the earlier eLife panel and
the remaining inputs for the
[resident source comparison](../elife-resident-source-v1#reproduce).
Append these arguments in the Slurm jobs and repeat with `--reverse`.

```sh
--matched-source-projection EARLIER_ELIFE_PROJECTION \
--matched-source-manifest EARLIER_ELIFE_MANIFEST \
--matched-source-supplement INITIAL_ELIFE_PROJECTION \
--matched-source-supplement-manifest INITIAL_ELIFE_MANIFEST
```
