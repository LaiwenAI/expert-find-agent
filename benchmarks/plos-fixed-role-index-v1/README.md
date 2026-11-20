# Allocate a fixed role index using historical topic exposure

The [coverage intervention](../plos-declaration-coverage-v1) revealed role
declarations at the top of each query ranking. Here a single index must serve
every query. Construct it before ranking the evaluation requests, then compare
its useful coverage with equally sized workload-agnostic indices.

Extract historical topic strings occurring on at least four articles from the
[pinned pilot](../plos-topic-role-postings-v1). Rank the author-disjoint target
articles for these historical strings, using only their titles and abstracts.
For each article, use weights of `1/rank` in the first eight positions. Build
two nested fixed inventories. Exposure ranking sums the weights. Balanced
allocation greedily increases the sum over historical topics of
`log(1 + total selected rank weight for the topic)`. The objective gives
diminishing returns to topics already covered by the index. Resolve ties by
DOI. Compare with 12 independently salted DOI-hash inventories. At each budget
of 20 or 39, select the corresponding prefix of each inventory. The target
articles' publisher topics and contributor roles are excluded from allocation.

Evaluate the same frozen 46 requests from the
[author-disjoint catalog](../plos-topic-role-holdout-v1), reporting exact topic
strings absent from the historical workload separately. This is a follow-up on
an already explored evaluation catalog. Measure declaration coverage among
the first four candidates, numbers of sources read, and the original publisher
tag and owner based recommendation counts. The four- and eight-source topic
controls must reproduce the published aggregate counts. All variants use the
same model, verifier cutoff and role bindings. Equal ordered batches share
scores within a query. XML bytes at the pinned digests measure how much raw
input each index needs for a full-XML build. They exclude the title and abstract
index and do not measure real parsing, partial fetch, or online service latency.

## Results

Slurm jobs `43749` and `43751` on an RTX 5090 reproduced all counts.
The fixed inventories use 59 historical topic strings. The target has 24
requests on novel strings and 22 requests on exact historical strings.
The table uses top-eight person nomination with up to four reads. Hash
columns show a median over 12 fixed inventories, with the minimum and
maximum in brackets. Coverage counts the first four slots for every
request, including repeat appearances of a source.

| Panel and budget | Method | First-four slots with declarations | Tagged / off-label recommendations | Reads |
| --- | --- | ---: | ---: | ---: |
| Novel, 20 | Exposure | 34 / 96 | 22 / 2 | 96 |
| Novel, 20 | Balanced | 32 / 96 | 22 / 2 | 96 |
| Novel, 20 | Hash | 25 [10–44] / 96 | 22 [20–23] / 2.5 [2–6] | 96 [94–96] |
| Novel, 39 | Exposure and balanced | 50 / 96 | 22 / 4 | 95 |
| Novel, 39 | Hash | 50 [30–60] / 96 | 22 [21–24] / 4.5 [2–8] | 95 [88–96] |
| Historical, 39 | Exposure and balanced | 76 / 88 | 20 / 2 | 70 |
| Historical, 39 | Hash | 42 [28–54] / 88 | 19 [17–20] / 2 [2–2] | 85 [82–87] |

For novel topics, the pure topic baselines obtain 20 tagged and 2
off-label recommendations at four reads, and 24 tagged and 10 off-label
at eight. At budget 20 the fixed workload policies gain two
publisher-backed slots against topic four with the same read count,
matching the hash median. The larger trained inventory does not improve
that count. On historical strings, the 39-source inventories save
15 reads against the hash median in the bounded arm and obtain one
more tagged recommendation. Whole-ranked person nomination uses all
four reads and obtains the same tagged counts as bounded nomination
for these learned inventories. With 39 declarations it emits an
additional off-label citation on the historical panel.

At 20 sources, the exposure and balanced inventories require 3.71 and
3.86 decimal MB of pinned XML respectively. The hash inventories range
from 3.76 to 4.53 MB. At 39, both historical policies require 7.84 MB
against a hash range of 7.61–8.37 MB. Byte counts are a build-input
check, and the observed reductions in reads are confined to the
historical workload in this comparison.

The stronger oracle coverage result from the previous experiment does
not transfer to novel topics with these ranking-only allocation
policies. The hash ranges describe this replay and these 12 seeds.
They are not sampling confidence intervals or a human relevance
assessment. Results motivate keeping a workload-agnostic control and
explicitly evaluating topic shift before relying on query history for
declaration placement.

[Aggregate results](results.json)

## Reproduce

Rebuild both frozen projections using the
[holdout protocol](../plos-topic-role-holdout-v1). Run on one Slurm CUDA
allocation with the original verified checkpoints and byte counts from the
[pinned XML files](xml-bytes.json).

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.fixed_role_index PILOT_PROJECTION HOLDOUT_PROJECTION \
  benchmarks/plos-topic-role-holdout-v1/manifest.json QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/plos-fixed-role-index-v1/xml-bytes.json \
  benchmarks/plos-bounded-topic-v1/results.json NEW_RESULTS.json
```
