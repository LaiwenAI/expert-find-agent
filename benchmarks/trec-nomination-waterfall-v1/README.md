# Separate expert nomination, evidence reach and document reads

Use the 49 judged [TREC 2006 Enterprise topics and two qrels](https://trec.nist.gov/data/t15_enterprise.html).
Discovery uses the original title and description and a positive
support graph built from the other topics. TF-IDF cosine ranks
historical topics. Each contributes reciprocal-rank weight to its
historical associations. For every held topic the runner rebuilds
the graph with that topic removed and verifies unchanged nominations
and source order. A positive slot requires grade 2 for both the
expert and the cited document supporting that person.

| Strategy | Candidate and source selection |
| --- | --- |
| Document-first | Keep the first 256 historical documents. Rank experts by their best three associated document scores. |
| Expert-first | Give each distinct expert one vote per historical topic, regardless of that person's document count. Rank associated sources directly for each expert. |
| Rank fusion | Combine the two top-20 rankings by reciprocal rank and use the expert-first source lists. |

## Fixed candidate–document checks

At 13 checks per topic, all arms spend 637 checks. Keep the first 20
nominees and examine the first source of each nominee in order until
the budget is exhausted.

| Strategy | Known supported experts with accessible evidence anywhere | Accessible in first 20 | Supported recommendation slots | Distinct query–document reads |
| --- | ---: | ---: | ---: | ---: |
| Document-first | 195 | 30 | 6 | 108 |
| Expert-first | 421 | 153 | 8 | 590 |
| Rank fusion | 421 | 126 | 6 | 585 |

Counts sum across topics. Of 1,287 expert–topic positives with a
known grade-2 citation, 421 have any such pair in the historical
positive graph. Within the expert-first shortlist, just 18 of the
153 accessible positives retain any positive citation at the first
source position. The figure increases to 76 in the first 16 positions.
Candidate nomination, source ordering and shared document reads
therefore affect different stages.

## Cap reads and checks together

Keep up to 16 sources per nominee and 13 candidate–document checks.
At a common maximum of two, three or five distinct documents, compare
checking the next ranked candidate's source with prioritizing already
opened documents. For a new document, the reuse policy sums
candidate-rank and source-rank discounts across eligible linked people.
It receives no held relevance labels before choosing a check. An
unknown pair remains unconfirmed.

The strict comparison then restricts every alternative to the **actual
number** of documents read by the document-first reference on that same
topic under the three-read limit. Candidates can exhaust their eligible
actions before they reach 13 checks. The ceiling selects up to that
many documents with full access to the labels, maximizing the number
of known supported expert slots within the fixed top-20 and source
menus. The ceiling has no routing or model cost attached.

| Strategy and scheduler | Reads | Checks | Supported slots | Ceiling |
| --- | ---: | ---: | ---: | ---: |
| Document-first, rank-first | 104 | 637 | 6 | 24 |
| Expert-first, rank-first | 104 | 171 | 3 | 49 |
| Expert-first, reuse-first | 103 | 285 | 9 | 49 |
| Rank fusion, reuse-first | 103 | 363 | 5 | 47 |

The expert-first reuse policy's paired supported P@5 difference
against the reference is +1.22 percentage points, with a
topic-bootstrap interval of −3.27 to +6.53. It uses fewer checks and
one fewer read in this replay. The interval leaves its quality gain
unconfirmed. The 49-slot ceiling and nine recovered slots identify
room to improve source allocation even within the retrieved menus.
These are archived-graph and human-judgment replay results.
They do not include W3C corpus retrieval, source opening latency,
live verification, or new labels for unjudged associations.

Both reverse-order Slurm CPU replicas reproduce each report. The
matched-read rerun also reproduces the intermediate common-cap table.
[Aggregate results and both digests](results.json)

## Reproduce

Restore the official topics and qrels, then run in a Slurm CPU allocation.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m experiments.trec_enterprise.nomination_waterfall \
  TREC_DATA CHECK_BUDGET_RESULTS
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m experiments.trec_enterprise.nomination_read_budget \
  TREC_DATA SOURCE_BUDGET_RESULTS
```

Use `--reverse` to replicate the topic iteration order check.
