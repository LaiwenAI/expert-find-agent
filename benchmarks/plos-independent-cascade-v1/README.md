# Transfer the fixed legacy-note verification cascade

Freeze the numerical cutoffs and gate features recorded in the [existing cascade result](../plos-legacy-note-cascade-v1/results.json). Download legacy PLOS ONE notes from the publisher's 2010–2015 inventory. Require titles containing a traffic, transport, hydrology or water-related search phrase. Select up to 64 eligible papers by a registered hash order after excluding the prior capture's DOI inventory. Exclude a paper if its normalized byline names overlap any downloaded paper used in the earlier capture. Include one original two-duty contrast per retained paper, with unmodified contribution text and the same test phrasing used by the previous experiment. The [capture receipt](capture.json) records all selected DOI hashes prior to applying the attribution filter.

Verify all five frozen routes. The reference uses a 4B reader for every paired request. The character and semantic gates use the same fixed acceptance, rejection and fallback choices from the original calibration. Eager and lazy agreement execute the same decision policy. Exactly preserve both outputs and mandatory reader calls when changing its schedule.

Run five warmed, sequential repetitions per article and rotate the order of arms across observations. Require repeatable verdicts and model-call counts. Report the median time per article and derive the mean and P95 from those medians. Include scoring, bound-clause lookup and the reader call when needed. Exclude index construction, source discovery, request transport and model load. Paired uncertainty for support, wrong author and mean latency resamples connected groups of shared normalized author names. Counts reflect publisher-declared duties, not general expertise or arbitrary narrative evidence.

## RTX 5090 results

Job 42636 retains 45 of 64 downloaded papers across 44 author-name groups. Six papers lack a usable note or license marker and 13 lack a two-duty contrast. None of the retained notes shares a normalized byline name with the old capture. All warmed repetitions agree for each route. Read and vector call counts below refer to one pass over the papers.

| Route | Supported accepted / 45 | Wrong owner accepted / 45 | Both correct | 4B calls | Vector calls | Mean | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Always 4B | 33 | 8 | 28 | 45 | 0 | 21.38 ms | 24.58 ms |
| Character gate | 40 | 7 | 34 | 28 | 0 | 13.47 ms | 24.77 ms |
| Semantic gate | 35 | 12 | 24 | 4 | 45 | 10.10 ms | 28.38 ms |
| Agreement, eager | 36 | 4 | 32 | 35 | 45 | 24.98 ms | 32.88 ms |
| Agreement, lazy | 36 | 4 | 32 | 35 | 26 | 21.59 ms | 32.80 ms |

Relative to always reading, the character gate has a +15.56 pp paired support difference (95% author-group interval +2.22 to +29.55 pp), a −2.22 pp wrong-owner difference (−13.64 to +8.89 pp), and a −7.91 ms mean latency difference (−11.01 to −4.99 ms). Its P95 remains close to the reference. The earlier energy panel observed more wrong-owner acceptances under the same gate, so the observed risk advantage is cohort-dependent.

The two agreement schedules reproduce all 45 decisions and required reader calls. Lazy scheduling skips 19 of the 45 vector forwards. The agreement gate reduces wrong-owner acceptances by four against always reading (paired 95% interval −18.18 to −2.17 pp). Its support difference spans zero and its P95 is 8.22 ms higher. The 4B route itself also misattributes some evidence on this transfer panel.

This is stronger independent support for using a cheap lexical gate on this note format and for avoiding redundant execution under an unchanged agreement policy. It neither demonstrates lower service P95 nor establishes an attribution-risk guarantee outside the tested format. [Aggregate results](results.json)

## Reproduce

Restore both captures from the released receipts. Run with the same frozen open checkpoints as the initial cascade, on a Slurm GPU node.

```sh
PYTHONPATH=src:. python -m experiments.credit.legacy_contribution_text restore OLD_CAPTURE OLD_SNAPSHOT
PYTHONPATH=src:. python -m experiments.credit.legacy_cascade_transfer restore TRANSFER_CAPTURE NEW_SNAPSHOT
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.legacy_cascade_transfer run \
    OLD_SNAPSHOT NEW_SNAPSHOT QWEN_EMBEDDING QWEN4B ORIGINAL_CASCADE_RESULTS NEW_RESULTS
```
