# Transfer same-person role screening to a disjoint publisher catalog

The [pilot](../plos-topic-role-postings-v1) used a dense and
lexical topic ranker, followed by a frozen reranker and
publisher-declared role binding. Transfer its topic acceptance
threshold of −2.0625 without retuning. The new catalog uses
a [frozen public manifest](manifest.json) from three previously
captured PLOS source collections. Exclude pilot DOIs, exact
case-folded titles, validated ORCID overlaps, and normalized
author-name overlaps among the searchable pilot records.
All admitted articles have a title, abstract, exact leaf topics,
and at least one declared role for every author. Author names
and ORCIDs are withheld from the output. The sources come from
the same publisher and were available for other investigations.

As in the pilot, take topics found in at least four searchable
articles and form one rare and one frequent two-role request
per eligible topic. The publisher's topic tags supply the
evaluation reference and stay outside retrieval and verification.
The separate novel-topic panel uses tag strings absent from the
pilot's eligible topic list. Some themes or articles may recur
across related tag strings within this evaluation.

## Frozen comparisons

All policies use the same article order and reranker. Hash masks
define roughly half and complete declaration availability.

| Route | Article selection for one verifier batch |
| --- | --- |
| Topic eight | Best eight by topic |
| Topic four | Best four by topic |
| Union four | First four compatible articles using collective role unions |
| Person four | First four compatible articles using same-person role postings |
| Compact eight | Remove proven nonmatching articles from the original eight |
| Adaptive | Start with compact eight. When fewer than four remain, fill to at most four from deeper compatible articles |

An unindexed declaration remains eligible under every route.
Compact eight preserves every original source that could provide
both roles under the available declarations. It still executes a
smaller neural batch, which can affect numerical scores. After
topic verification, role checks read original declarations and
return up to five deduplicated people with their supporting papers.

The primary transfer comparison is Person four against Topic
eight at complete declaration coverage. Freeze the following
criteria before running the GPU experiment.

- The 95% paired interval lower endpoint for grounded P@5 must
  be at least −1 percentage point, equivalent to tolerating one
  missing grounded result per 20 five-slot queries.
- The corresponding upper endpoint for off-label citations per
  five slots must be at most +1 percentage point.
- The 95% paired endpoint for P95 latency change must be below
  zero at its upper end.

Resample whole topic strings; articles can occur under multiple
strings. These intervals are descriptive for the sampled topic
catalog. Off-label results may be relevant despite missing tags.
The half-coverage, novel-topic, compact, and adaptive results
are secondary. Also report equal four-source budgets and exact
witness-list differences. Timing covers query encoding and local
module execution. It excludes remote I/O, queues and serving.

## Results

Job `43470` completed on one RTX 5090. The frozen manifest
contains 78 articles. The test uses 46 requests on 23 topic
strings, 12 absent from the pilot's eligible list. The partial
index contains 42 of the 78 declarations.

| Indexed declarations | Route | Grounded / 230 | Off-label | Reads | P95 | Witness lists changed vs Topic eight |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 78 | Topic eight | 49 | 14 | 368 | 64.2 ms | 0 |
| 78 | Topic four | 37 | 4 | 184 | 34.5 ms | 15 |
| 78 | Union four | 42 | 10 | 184 | 37.6 ms | 14 |
| 78 | Person four | 52 | 16 | 184 | 37.6 ms | 18 |
| 78 | Compact eight | 49 | 15 | 161 | 46.3 ms | 2 |
| 78 | Adaptive | 57 | 21 | 220 | 46.4 ms | 9 |
| 42 | Topic eight | 49 | 14 | 368 | 63.6 ms | 0 |
| 42 | Person four | 41 | 7 | 184 | 37.3 ms | 13 |
| 42 | Compact eight | 49 | 14 | 251 | 55.3 ms | 1 |
| 42 | Adaptive | 50 | 14 | 263 | 55.0 ms | 2 |

**Primary transfer.** Person four reduces P95 by 26.6 ms
relative to Topic eight. The paired 95% interval is −29.8 to
−22.0 ms. Grounded P@5 changes by +1.30 points, with a 95%
interval of −3.04 to +6.09. Off-label citations per five slots
have an interval of −1.30 to +3.48 points. The speed criterion
passes and both predeclared quality criteria fail. At equal
four-source budgets, Person four gains 6.52 grounded P@5
points (95% interval +2.61 to +10.87). The corresponding gain
with 42 indexed declarations is +1.74 points (0 to +3.91).

**Coverage-aware follow-up.** With 42 indexed declarations,
Compact eight matches the original grounded and off-label
counts, reduces reads by 31.8%, and reduces P95 by 8.3 ms
(95% paired interval −12.4 to −1.6 ms). One ordered witness
list changes. At full coverage, two lists change and one
additional off-label citation causes the upper risk bound to
exceed +1 point. Adaptive improves grounded coverage at full
coverage and increases off-label citations. These results
prioritize verifying batch stability and topical confirmation
before using a shorter evidence budget in enterprise search.

[Full results, including the held-out topic subset](results.json).

## Reproduce

Rebuild the [pilot projection](../plos-topic-role-postings-v1)
using the four earlier raw collections. The candidate folders are
`plos-source-gate-holdout-v1`, `plos-source-gate-holdout-secondary-v1`,
and `plos-credit-source-scoped-v1`; use raw publisher copies with
the frozen digests. Run the replay in a Slurm GPU allocation.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.topic_role_holdout prepare \
  PILOT_PROJECTION NEW_PROJECTION NEW_MANIFEST \
  --raw-folders PILOT_XML_DIR [MORE_PILOT_XML_DIRS...] \
  --candidate-folders CANDIDATE_XML_DIR [MORE_CANDIDATE_XML_DIRS...]
python -m experiments.credit.topic_role_holdout run \
  NEW_PROJECTION benchmarks/plos-topic-role-holdout-v1/manifest.json QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
