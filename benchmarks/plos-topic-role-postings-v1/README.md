# Find experts for an open topic with two declared roles

Search for people credited with both requested CRediT roles
on any publication assigned an exact requested topic by PLOS.
Each request gives a topic and two canonical roles. Any
publication meeting those requirements can be the witness.

Rebuild the source projection from the pinned publisher XML
in the [role-posting fixture](../plos-credit-role-postings-v1).
Verify each XML digest, DOI and declared role list. Keep the
title and abstract for retrieval. Withhold the leaf subjects
from `Discipline-v3` at every ranking and verification stage.
Article-scoped author ordinals identify contributors when
there is no verified publisher identifier. Deduplicate
across articles only with the fixture's publisher-identifier
tokens. The returned witness must itself carry both roles
for the same person.

Retain topics occurring in at least four searchable papers.
For each evaluation topic, use two role pairs selected by
their publisher-declared coverage, one rare and one more
frequent. Each has support in at least one but fewer than
all tagged articles. Hold out whole topic strings by hash
for calibration. The catalog and some articles are shared
between topics.

All policies rank titles and abstracts by reciprocal-rank
fusion of exact dense similarity and IDF lexical overlap.
Open at most two, four or eight articles and run the same
frozen reranker on their title and abstract in one batch.

| Policy | What can remove an article before verification |
| --- | --- |
| Topic first | Topic rank budget only |
| Role union | A fully indexed article lacks at least one of the two roles across all authors |
| Person postings | A fully indexed article lacks a single person credited with both roles |

In both posting arms, articles with incomplete or masked
index entries stay eligible. Mask coverage uses a stable DOI
hash. After verification, all arms use the same original
publisher declarations to bind roles. The reranker sees only the topical query, title and abstract. Freeze its
threshold on the top eight articles for each calibration topic
at a 10% observed off-label acceptance ceiling.

Report reachable requests, at most five returned identities,
and accepted recommendations citing a witness that satisfies
the publisher topic label and both same-person role fields.
A publication missing the exact topic tag may still be
relevant. Off-label citations retain unresolved topical
relevance. The role fields define the
structured role target and are used to implement the index.
The withheld topic metadata tests whether the routed
publications satisfy a separate signal. The timing includes
query encoding, source scoring, selection and live reranker
but omits publisher downloads, index construction and service
queues. Reuse the measured query-only latency across arms,
rotate verifier execution order, and bootstrap by topic.

## Results

Slurm jobs `43462` and `43465` reproduced all outcome and
read counts on one RTX 5090. Use 147 searchable articles
after excluding five without a released title-and-abstract
pair. Two retained articles have no leaf topic tags. The
nominal half and full masks index 76 and 146 source
declarations. Calibration uses 19 topic strings. At the
frozen cutoff, the reranker accepts 9/93 off-label articles
and 27/59 matching articles. Evaluation has 80 queries on
40 other topic strings.

| Indexed declarations | Source limit | Routing | Reachable / 80 | Grounded / 400 | Off-label citations | P95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 146 | 4 | Topic first | 40 | 63 | 25 | 33.2 ms |
| 146 | 4 | Role union | 53 | 70 | 30 | 45.0 ms |
| 146 | 4 | Person postings | 64 | 84 | 31 | 45.2 ms |
| 76 | 4 | Topic first | 40 | 63 | 25 | 32.8 ms |
| 76 | 4 | Role union | 46 | 65 | 29 | 33.4 ms |
| 76 | 4 | Person postings | 50 | 69 | 29 | 44.3 ms |
| Shared | 8 | Topic first | 54 | 83 | 30 | 81.2 ms |

Each four-source route inspected exactly 320 articles.
With 146 indexed declarations, person postings add 5.25
points in fixed-denominator grounded P@5 versus topic first
(95% paired topic-bootstrap interval +2.75 to +8.25
points). The gain is 1.50 points (+0.25 to +3.25) when
76 declarations are indexed. At full coverage, the rare-duty
subset gains 11 grounded citations, from 10 to 21, versus
10 additional grounded citations in the frequent subset.

The exploratory four-versus-eight comparison gives 84 versus
83 grounded citations and 31 versus 30 off-label citations
at full coverage, with 320 versus 640 source reads. P95 is
45.2 versus 81.2 ms. The paired interval for grounded
P@5 difference is −1.75 to +2.50 points. With the half
mask the interval is −5.75 to −1.50 points, indicating
a loss at the narrower budget in this panel. A future
noninferiority comparison needs a predefined quality margin
and an independent catalog. The current evidence supports
early same-person screening when declarations are explicit
and coverage is high. [Aggregate results](results.json)

## Reproduce

Prepare the [pinned CC BY source snapshots](../plos-credit-role-postings-v1).
Run the second command in a Slurm CUDA allocation with the
verified open checkpoints, and use fresh output paths.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.topic_role_postings prepare \
  benchmarks/plos-credit-role-postings-v1/sources.json NEW_PROJECTION XML_DIR [MORE_XML_DIRS...]
python -m experiments.credit.topic_role_postings run \
  NEW_PROJECTION QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
