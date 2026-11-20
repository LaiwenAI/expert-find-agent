# Transfer the eLife first-query policy to PLOS at matched reads

Reconstruct the 78-article PLOS catalog from the published
[holdout manifest](../plos-topic-role-holdout-v1). The holdout
excludes records overlapping the earlier PLOS pilot by exact title
or listed author identity. The test uses both two-role requests
for every eligible leaf topic. Also report exact topic strings
absent from the earlier PLOS pilot's topic list. Multiple topics
can reference some of the same documents or describe related
concepts.

Transfer the BGE-small hybrid ranker, the MiniLM L12 verifier,
and the relevance cutoff from the previous
[eLife factorial study](../elife-first-query-factorial-v1).
Verify the checkpoint file hashes. No threshold or policy tuning
uses PLOS outcomes. Run the same five routes as the
[eLife matched-read replay](../elife-first-query-equal-reads-v1):
four initial reads, an eight-source local pool and two fixed or
adaptive extra reads. Each route sees the same title and abstract
fields and reads the original role declarations during verification.
Report sparse and full declaration masks separately.

The primary contrast is person-first, fixed local versus topic-first,
fixed local. All queries in that contrast verify exactly six unique
articles. The adaptive and fixed global comparisons measure the
cost of smaller realized local work and larger scope, respectively.
Report tag-backed and off-label slots out of a fixed five per query,
and paired percentile intervals resampling topic strings. Off-label
citations may be relevant. These are publisher-tag diagnostics and
do not validate expert fit for enterprise needs.

## Results

The panel has 46 requests across 23 topics. Twenty-four requests
use 12 exact topics absent from the earlier PLOS pilot. The table
shows tag-backed slots / off-label slots / total reads, out of a
fixed 230 recommendation slots.

| Route | 42 indexed declarations | 78 indexed declarations |
| --- | ---: | ---: |
| Topic, adaptive local | 48 / 12 / 264 | 51 / 12 / 240 |
| Person, adaptive local | 53 / 12 / 234 | 55 / 18 / 216 |
| Topic, fixed local | 48 / 12 / 276 | 51 / 12 / 276 |
| Person, fixed local | 53 / 12 / 276 | 55 / 18 / 276 |
| Person, fixed global | 54 / 14 / 276 | 56 / 19 / 276 |

At matched reads, the sparse mask gains 5 tag-backed slots with
the same off-label count. The paired difference is +2.17 percentage
points in tag-backed P@5 (95% interval +0.43 to +4.35).
The full mask gains four tag-backed and six off-label slots,
with intervals 0.00 to +3.91 and +0.43 to +5.22 points, respectively.
The novel-topic subset gains one tag-backed slot at either coverage,
with both intervals including zero. Global fixed rescue adds one
tag-backed slot at both coverages and no net tag-backed gain on novel
topics under the full mask.

Despite fewer realized reads under the adaptive policy, the original
run's warm serial P95 is 476 ms for person-first versus 464 ms for
topic-first under the full mask. The primary pair reproduced all non-timing counts when replayed
with its execution order reversed. At fixed reads and full coverage,
person-first P95 was 497–504 ms versus 481 ms for topic-first.
The measured replay includes query
encoding, ranking, and both verifier stages, with one worker on an
Apple M4. It excludes setup, external input/output and queuing. This
transfer supports early person-based nomination for some requests
and shows that the serial speedup and the evidence mix vary by catalog.

[Aggregate results and intervals](results.json)

## Reproduce

Rebuild the PLOS projection as described in
[the holdout benchmark](../plos-topic-role-holdout-v1).
Use the BGE-small and FlashRank checkpoints from the eLife benchmark.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.role_first_query_transfer PLOS_PROJECTION \
  benchmarks/plos-topic-role-holdout-v1/manifest.json \
  benchmarks/elife-first-query-factorial-v1/results.json \
  BGE_MODEL_DIRECTORY FLASHRANK_CACHE_DIRECTORY NEW_RESULTS
```

For the primary-only repeat, set `live.ARMS` to
`("topic_fixed", "person_fixed")` and `COMPARISONS` to
`(("person_fixed", "topic_fixed"),)` in a caller of
`role_first_query_transfer.run`. Use order offset 1.
