# Verify author scope and activity separately on fixed windows

Reuse the article cohorts, candidates, query phrases, and selected
windows of the [long-article study](../plos-full-article-role-windows-v1).
For each query, independently freeze the evidence selected by the
owner postings and contribution-section selectors. For a given
selector, no verifier can request a different window.

| Verifier | Author binding | Semantic comparison |
| --- | --- | --- |
| Joint | Frozen Qwen3-Reranker-0.6B receives the candidate initials | Same cross-encoder scores whether the original window supports the requested activity for that person |
| Scoped joint | Check the selected window for a complete `activity: INITIALS.` declaration with the exact candidate | Apply the same joint instruction only to passing windows |
| Scoped role | Same structural check | Remove candidate initials from the query, compare the requested activity to the same original window |
| Scoped head | Same structural check | Compare the requested activity to the exact original action head of that window |
| Scoped joint, frozen threshold | Same structural check and scores as scoped joint | Reuse the joint baseline's calibrated threshold to isolate enabling the scope check from recalibration |

The structural check never looks at the publisher section identifier
or role label. It may abstain when there is no parseable local author
list. All neural variants use the same pinned model. Score passing
candidate–window pairs in one batch, and avoid the model if both fail
the scope check. Timing covers scope checking, query preparation and
model calls on selected windows. It excludes choosing the windows,
document retrieval, index construction and service queueing.

The 30 calibration papers determine each selector and verifier's
threshold at the earlier study's 10% cap for papers with any
unsupported acceptance. Thresholds are fixed for the 20 earlier test
and 45 transfer articles. A further stress test uses the same 45
transfer articles and candidate pairs, with non-credit administrative
requests drawn from the [existing activity controls](../plos-anchored-role-margin-v1).
Selectors run on these requests before the windows are frozen for the
verification comparisons. Those phrases do not enter calibration.
Count mismatched authors, wrong activities on an owned declaration,
and accepted passages outside the reference declarations separately.
The labels establish what these publisher notes declare.

## Results

Slurm job 43101 used one RTX 5090 on ws5. The joint rows reproduce
all corresponding long-article outcomes, including selected
positives, accepted positives, adverse papers, and partial authors.
Each result below is clean papers / adverse papers. A paper may
contain more than one accepted unsupported attachment.

| Selector | Verification | Earlier test (20) | Transfer (45) | Transfer reader pairs | Transfer verifier P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Contribution section | Joint | 11 / 3 | 20 / 10 | 90 | 8.72 |
| | Scoped joint at frozen threshold | 11 / 3 | 21 / 9 | 44 | 8.34 |
| | Scoped joint, recalibrated | 11 / 3 | 21 / 10 | 44 | 8.34 |
| | Scoped role, original window | 11 / 4 | 24 / 11 | 44 | 7.41 |
| | Scoped head | 11 / 0 | 24 / 0 | 44 | 7.35 |
| Author postings | Joint | 6 / 0 | 7 / 0 | 90 | 8.74 |
| | Scoped joint at frozen threshold | 6 / 0 | 7 / 0 | 90 | 8.74 |
| | Scoped joint, recalibrated | 6 / 0 | 7 / 0 | 90 | 8.74 |
| | Scoped role, original window | 2 / 0 | 0 / 0 | 90 | 8.72 |
| | Scoped head | 9 / 4 | 23 / 9 | 90 | 8.67 |

On the contribution-section transfer windows, structural binding alone
with the original threshold removes the one wrong-owner acceptance.
Ten wrong-role attachments to correctly bound authors remain on nine
papers. Binding and scoring only the action head gives 24 clean
papers, compared with 20 for the joint verifier, with a descriptive
author-cluster 95% interval of +2.17 to +18.18 percentage points.
All 24 correct windows are accepted. The 44 comparisons replace 90,
and verifier-only P95 falls by about 16%. All four primary verifiers
abstain across the 45 administrative requests for both selectors.

On the author-postings windows, scoring the head accepts all 28
correctly selected clauses but also accepts nine wrong-role clauses
on nine papers. The additional challenge is semantic, even after
the candidate is bound. The two selectors expose different sets of
role foils, so scope and head scoring must be evaluated with a
selector-specific risk check. The favorable section result is an
exploratory finding with recurring phrase templates and strict
publisher syntax; another format and fresh wording are needed
before applying it to nonstandard or enterprise evidence.

[Aggregate results](results.json)

## Reproduce

Use the pinned captures and reranker from the
[long-article study](../plos-full-article-role-windows-v1)
on a Slurm CUDA compute node.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.factored_article_role_verification \
    OLD_PUBLISHER_SNAPSHOT TRANSFER_PUBLISHER_SNAPSHOT RERANKER_CHECKPOINT NEW_RESULTS.json
```
