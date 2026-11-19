# Compare deeper owned evidence with role-first candidate retrieval

Use the 80 enterprise development requests from the
[fixed-budget replay](../enterprise-gap-budget-replay-v1).
Keep the original parsed criteria and existing recommendations.
No sealed query is used.

Build an inverted index over bounded original passages in
development profiles. Use explicit project roles and inventor lists
from the earlier parser. Add biography clauses only when the
candidate's exact name starts the clause and is followed by a
recognized contribution verb. Mere mentions and coauthorship do
not become ownership signals. These are profile claims and
require human verification.

| Selector | Candidate and evidence policy |
| --- | --- |
| Frozen pool | Four best static actions from the prepared original pool |
| Deeper owned | Search the original unresolved candidates' bounded role windows across the full development inventory |
| Global roles | Search the same index for both existing and new candidates, excluding candidates already issued by the original pipeline |

The two new selectors retrieve windows that have lexical overlap
with the objective and prioritize the weakest and average criterion
coverage with an original-rank prior. They read no more than two
windows per candidate. Unlike the frozen control, they only use
bounded passages, so a broader index can still yield fewer eligible
options for an original candidate list. All methods read at most
four windows per request. A provisional promotion requires a
positive verifier margin for all criteria in a single window. The
same pinned Qwen3-Reranker-0.6B scores each window with a live forward.

Report both the full four-window cap and a paired restriction that
truncates all selectors to the smallest available count in each
query. The paired restriction conditions on all methods having an
available window. Full-cap coverage reports what each policy can
reach and does not equalize the realized model-pair counts. Profile
index build and retrieval timings are local preparation measures.
Reader timings come from Slurm GPU inference on preselected
windows and are not end-to-end times.

The private packet checks each possible promotion against its exact
original text offsets and writes two unfilled reviewer forms.
Aggregate results do not contain the private prepared index,
queries, profiles, outstanding decisions or reviewer labels.

## Results

Slurm job 43127 used one RTX 5090 on ws5. The index contains 5,734
role-bearing windows linked to 1,098 development experts. Project
lead or participant fields account for 5,665 windows. There are
60 explicit inventor windows and nine name-led biography windows.

| Budget view | Selector | Requests with windows | Window / criterion checks | Provisional experts | First provisional results | Verifier P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Paired count | Frozen pool | 44 | 107 / 209 | 3 | 1 | 66.07 |
| | Deeper owned | 44 | 107 / 209 | 4 | 1 | 39.57 |
| | Global roles | 44 | 107 / 209 | 5 | 3 | 43.44 |
| Full cap | Frozen pool | 55 | 184 / 331 | 4 | 1 | 67.87 |
| | Deeper owned | 44 | 107 / 209 | 4 | 1 | 39.57 |
| | Global roles | 79 | 316 / 508 | 28 | 4 | 44.78 |

All five global promotions at the paired budget are outside the
original candidate lists. At the full cap, 27 of 28 promotions
are outside those lists. The 28 slots come from 20 requests,
including four unanswered requests. The named biography rule
accounts for just one global window among all full-cap neural checks.
Local in-memory retrieval P95 is 0.44 ms for deeper owned and
10.65 ms for global roles; do not add these independently measured
quantiles to the GPU measurements.

The method-blind packet contains 33 distinct provisional views and
zero human judgments. The retrieval route shows that usable role
evidence can be found outside the initial shortlist at matched
verification counts. The quality comparison still requires
adjudication of expertise, hard constraints, and personal
contribution. The higher full-cap coverage uses more verification
pairs. The matched comparison excludes any request with no
eligible window in at least one arm.

[GPU aggregate](results.json)

## Reproduce

Use the authorized development dataset, the frozen role panel and
the routed retrieval trace from the previous experiment. Run the
middle command on a Slurm CUDA node.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_role_pool_expansion prepare \
  DATASET PRIVATE_FROZEN_PANEL PRIVATE_ROUTED_TRACE PRIVATE_PREPARED_PANEL
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_role_pool_expansion run \
    PRIVATE_PREPARED_PANEL RERANKER_CHECKPOINT NEW_RESULTS.json PRIVATE_NEW_TRACE
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_role_pool_expansion packet \
  DATASET PRIVATE_PREPARED_PANEL PRIVATE_NEW_TRACE PRIVATE_REVIEW
```
