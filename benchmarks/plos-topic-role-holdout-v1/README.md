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
