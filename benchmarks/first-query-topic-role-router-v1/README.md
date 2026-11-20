# Freeze a cheap gate for early role nomination

Use the matched-read eLife and PLOS tasks, quantized BGE-small and
MiniLM L12 checkpoints, and the already frozen eLife verifier cutoff.
Compare the two exact six-read local routes from the
[matched-read study](../elife-first-query-equal-reads-v1).
Choose between those routes before opening either verifier batch.
Both arms leave unknown declarations eligible. They check the
actual source declaration and topical relevance after reading.

The gate uses only request words, ranked titles, and indexed
role postings. Count explicitly incompatible records in the first
four topical positions. Inspect the replacement candidates in the
role-aware first four. The gate may switch only if one newly
introduced candidate has a known same-person role declaration and
meets a topical rank limit. Compare no title-overlap criterion,
at least one shared title word, and all request topic words occurring
in the title. The gate can default to topical nomination for all cases.

Fit on the original eLife development topic strings, pooling panel and
visibility settings. Compare 36 fixed switches plus the always-topic
fallback. Require at most the baseline off-label slot count, then
maximize tag-backed slots, breaking ties by fewer off-label slots
and fewer switches. The source mask, topic labels, and verifier outputs
from either test set do not enter the fit. Apply one frozen rule to the
remaining eLife topics and the entire pinned PLOS holdout.

Cache scores only for an identical query and ordered source batch.
This paired replay uses the same input batch for each reused score.
Report six article reads per query, label counts, and paired intervals
by exact topic string. Timings are out of scope. Off-label answers
remain unadjudicated and serve as a development proxy.

## Results

Forty development rows cover 20 distinct two-role requests on
10 topic strings, each repeated under two index masks. All 37
rules meet the development off-label limit. None improves tag-backed
slots against always-topic. The tie-break
selects the always-topic route (30 tag-backed and 10 off-label
development slots).

The table gives tag-backed / off-label slots. All routes read
six sources per request. The frozen router matches the first
column in every test cell.

| Test panel | Topic only / frozen router | Always use early person nomination |
| --- | ---: | ---: |
| eLife headings, sparse and full (24 requests per mask) | 5 / 2 | 5 / 2 |
| eLife keywords, sparse (36) | 49 / 19 | 49 / 20 |
| eLife keywords, full (36) | 49 / 21 | 51 / 22 |
| PLOS leaf topics, sparse (46) | 48 / 12 | 53 / 12 |
| PLOS leaf topics, full (46) | 51 / 12 | 55 / 18 |

The gate does not reproduce the early-nomination lift. At full
coverage on PLOS it emits six fewer off-label slots than always
switching; at sparse coverage it gives up five tag-backed slots
with the same off-label count. The latter router-minus-person
paired interval is −4.35 to −0.43 points. The table does not
establish a routing benefit. Development covers too few topics
to select a useful switch from these features. The test topics have
also appeared in earlier benchmark studies, so this compares
transfer of a frozen rule on known public panels. A prospective
test would require independent topic and constraint annotations.
Two Python hash seeds produce identical reports.

[Aggregate results](results.json)

## Reproduce

Restore the eLife and PLOS projections using their pinned benchmark
manifests. Supply the local BGE-small snapshot directory and the
FlashRank cache from the eLife factorial replay.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.role_first_query_router ELIFE_PROJECTION PLOS_PROJECTION \
  benchmarks/plos-topic-role-holdout-v1/manifest.json \
  benchmarks/elife-first-query-factorial-v1/results.json \
  BGE_MODEL_DIRECTORY FLASHRANK_CACHE NEW_RESULTS
```
