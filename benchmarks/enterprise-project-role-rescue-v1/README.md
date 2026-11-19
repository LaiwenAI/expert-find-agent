# Recheck unresolved expert candidates using role-bearing records

Reuse all 330 unresolved candidate slots from the whole-request arm of
the [frozen live enterprise development run](../enterprise-routed-review-v1).
Hold the five candidate positions and parsed requirements fixed.
Compare two policies that may each select at most two evidence views
from a candidate's records. Both score lexical overlap using Han
character bigrams and Latin alphanumeric tokens, normalizing by the
square root of the view's distinct token count.

| Policy | Pool |
| --- | --- |
| Whole source | The opening 800 characters of each owned record, including papers and profile fields |
| Role passage | Clauses of at most 520 characters from project and patent profile fields that carry a role marker |

The deterministic scope check recognizes exact project lead and
participant labels in delimited fields. It rejects mixed-role project
clauses and simple explicit negation. Patent inventor lists require
the exact normalized candidate name. Project roles in a candidate's
profile are claims in that profile. They are not independent publisher
records, and a role label alone cannot attest technical fit or a
specific contribution. Narrative variants can evade this narrow parser.

Recheck a selected view only if it has a recognized role marker. Apply
the frozen enterprise reranker and require support for the objective
and all parsed prior qualifications in that single view. Use the same
criterion limits, task and positive-margin rule across arms. The whole
source can carry its role and topical evidence in different clauses.
The blinded reviewer must decide whether the role and claimed technical
evidence actually align. Original recommendations are held fixed.
Selection may revisit an original source because the preceding trace
does not record all sources considered for unresolved slots.

## Results

Slurm job 43026 ran the comparison on an RTX 5090. Counts below are on
330 unresolved slots over the same 80 development requests.

| Policy | Selected views | Slots with a role cue | Actual role-bearing verification calls | New provisional slots | Previously unanswered requests with a new provisional result |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole source | 660 | 69 | 69 | 1 | 0 |
| Role passage | 191 | 106 | 191 | 3 | 1 |

The role pool exposes cues on more slots while choosing fewer total
views. Its 191 actual neural calls exceed the ordinary policy's 69
because the latter skips many selected views without a role cue. At
the same first-check budget on the 69 slots where both methods reach
a role-bearing view, the ordinary route has one provisional supported
slot and the role route has two. This matched comparison conditions
on both methods finding a usable view, and does not estimate gains on
the rest of the panel. Verifier-only slot P95 is 13.68 ms versus 17.65
ms. This excludes selection and the earlier search. It is not an
end-to-end latency comparison.

The four distinct promoted evidence views form a method-blind review
packet with two unfilled copies and exact source offsets. The one
additional provisionally answered request would bring coverage to 35
of 80 when appended to the original provisional recommendations.
No independent relevance or ownership judgments are available. The
result gives a candidate source action to test, with small observed
coverage gains at unequal verification cost. The sealed cohort
remains unused.

[Aggregate results](results.json)

## Reproduce

Use the original private development dataset and live trace from
the [routed-review run](../enterprise-routed-review-v1). The first step
can run locally. Run the second in a Slurm CUDA allocation with the
pinned Qwen3-Reranker-0.6B checkpoint. Keep private packets local.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_project_role_rescue prepare \
  PRIVATE_DATASET ORIGINAL_PRIVATE_TRACE PRIVATE_PROPOSALS
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_project_role_rescue run \
  PRIVATE_PROPOSALS PINNED_RERANKER NEW_PRIVATE_TRACE
PYTHONPATH=src:. python -m experiments.retrieval.build_project_role_review \
  PRIVATE_DATASET PRIVATE_PROPOSALS NEW_PRIVATE_TRACE PRIVATE_BLIND_PACKET
```
