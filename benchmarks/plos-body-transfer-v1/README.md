# Transfer bounded source confirmation to a fresh author cohort

Use the frozen dense and lexical-body cutoffs from the [open source and authorship benchmark](../plos-open-legacy-attribution-v1/README.md) and the [bounded body replay](../plos-body-confirmation-v1/README.md). The new PLOS ONE capture covers environmental engineering, land use, remote sensing and municipal waste. Select DOI records in deterministic hash order before parsing the XML. Keep the earlier corpus in the search inventory as distractors. Papers used as test queries have disjoint DOIs and normalized byline names from both earlier captures. Cluster linked bylines within the new cohort for paired uncertainty estimates.

The intended paper must be supported by the held abstract excerpt or up to eight IDF-ranked keywords. The catalog indexes only the title and abstract prefix for nomination. The lexical route can use nonduplicate words from the nominated paper's body if the dense route abstains. Recompute document frequencies on the enlarged catalog, while keeping all numeric cutoffs and the open embedding and author-verification checkpoints frozen. Only the first nominated source can qualify. Synthetic clues and publisher role labels evaluate recovery of an exact study and its credited authors.

Compare both routes with and without the intended paper in the catalog. Report sources and downstream authors separately. Report both queries with at least one supported author and clean successes, which require a nonempty recommendation with every recommended author supported. Count wrong authors on queries with correct sources separately from wrong-source recommendations after withholding. The paired replay latency sums measured retrieval, the optional lexical body check and author verification. It excludes preparation, serving, and concurrent load.

The absolute dense score is the deployed control for this transfer. [The Magnitude Mirage](https://aclanthology.org/2026.emnlp-main.517/) motivates a separate comparison with score-distribution confidence from the same retrieval pass. [Adaptive Retrieval for Reasoning](https://aclanthology.org/2026.acl-long.1734/) is a comparison for a later open-search experiment that can seek a different source. The action here is limited to confirming the current nomination.

## Fresh-cohort results

Slurm jobs 42669 and 42673 ran on an RTX 5090 on ws5. The third capture retrieved 99 papers, of which 84 entered the catalog and 72 supplied test queries across 70 author-name components. The catalog contains 197 sources and 17,757 body windows. Raw top-one nomination found the intended study on 70 excerpt and 57 keyword queries. The second run recorded the clean-success metric and reproduced every earlier decision count.

| Clue | Gate | Correct source, present | At least one supported author, present | Clean success, present | Wrong author from correct source, present | Wrong-source recommendation, present | Wrong-source recommendation, withheld | Factored P95, present / withheld |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Excerpt | Dense | 52 | 45 | 30 | 17 | 0 | 0 | 99.16 / 7.77 ms |
| Excerpt | Dense + lexical body | 64 | 57 | 37 | 22 | 0 | 0 | 99.16 / 8.09 ms |
| Keywords | Dense | 40 | 32 | 20 | 14 | 1 | 4 | 99.28 / 7.44 ms |
| Keywords | Dense + lexical body | 50 | 42 | 27 | 17 | 1 | 4 | 99.28 / 7.47 ms |

The frozen lexical fallback adds 12 supported excerpt queries and 10 supported keyword queries. It adds seven clean successes for each clue form, or 9.72 percentage points. The descriptive author-cluster paired 95% intervals for clean success are +2.86 to +16.90 points for excerpts and +2.86 to +16.67 for keywords. Wrong recommendations on the correct source also increase by five excerpt queries and three keyword queries. The source-withheld controls add no wrong-source recommendations in this sample. Four keyword recommendations from wrong sources already occur under the dense baseline and remain after the fallback.

Both present-view P95 values are identical to two decimal places in this factored replay. The fallback adds 12 and 13 calls to the 4B author verifier for excerpts and keywords, respectively. A loaded service could therefore have different tail latency. The keywords also have a larger nomination gap, with 15 intended sources outside the top one compared with two for excerpts. This fallback has no way to repair those misses. The source and author effects are specific to the exact-study task and constructed clues. [Aggregate results](results.json)

## Reproduce

Restore the two earlier captures as documented in the integrated benchmark. Restore the new capture from its pinned publisher receipt, or run the capture command to collect a new snapshot. Run inference within a Slurm GPU job.

```sh
PYTHONPATH=src:. python -m experiments.credit.legacy_cascade_transfer restore \
  benchmarks/plos-body-transfer-v1/capture.json NEW_SNAPSHOT
PYTHONPATH=src:. python -m experiments.credit.legacy_body_transfer run \
  OLD_SNAPSHOT FIRST_TRANSFER NEW_SNAPSHOT QWEN_EMBEDDING QWEN4B \
  OPEN_ATTRIBUTION_RESULTS BODY_CONFIRMATION_RESULTS CASCADE_RESULTS NEW_RESULTS
```
