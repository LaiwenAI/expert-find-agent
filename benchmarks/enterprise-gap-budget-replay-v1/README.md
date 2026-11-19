# Allocate a fixed role-evidence budget across unresolved experts

Replay the [enterprise role-passage comparison](../enterprise-project-role-rescue-v1)
on the same 80 development requests and 330 unresolved candidate
slots. Keep the initial ranking, parsed constraints, recommendations
and first-query implementation fixed. For each request, pool the
already prepared role-bearing source prefixes and bounded role
passages for those candidates. Deduplicate exact source offsets per
candidate. These options include role claims from profiles; they
are not independently adjudicated contributions.

Each policy reads at most four windows per request. Run a genuine
sequential Qwen3-Reranker-0.6B forward after each selection. The
one- and two-read observations are prefixes of the same policy run.
Each action checks the same complete criterion list for that query.
All policies deprioritize confirmed candidates and use the same
number of checks per query at each evaluated budget.

| Policy | Next action |
| --- | --- |
| Rank round robin | Give each expert in the original rank order one read before advancing its depth. At each expert, prefer stronger lexical coverage. |
| Static bottleneck | Choose the best weighted combination of minimum and mean lexical criterion coverage and original rank. |
| Adaptive gap | Begin with the same static score. After a failed check, boost windows that cover the still-missing criteria of a partially supported candidate. Demote a candidate whose best prior check found no supported criterion. |

A promotion requires an explicit role marker and positive verifier
scores on all criteria in the same original window. The adaptive
policy does not combine unrelated evidence across observations.
There is no label-based action tuning. Report distinct promoted
candidates, requests with a promotion, and first provisional
recommendations for previously unanswered requests. The timings
cover online action selection and sequential verification on a
precompiled pool, excluding baseline search, pool construction,
source loading, network and queues.

A separate local command binds promoted evidence to immutable
source offsets and produces two blank method-blind reviewer forms.
The review focuses on candidate fit, applicable prior constraints
and individual contribution supported by the displayed evidence.
The published result contains only aggregate counts, hashes and
timing. It does not include requests, profiles, citations or labels.

## Results

Slurm job 43120 ran on one RTX 5090 on ws5. In 25 of the 80
development requests, unresolved candidates have no eligible window
in the prepared pool. The model criterion-check count matches
between policies for every budget setting.

| Window cap per request | Total window checks / criterion checks | Round robin, provisional candidates | Static, provisional candidates | Adaptive, provisional candidates | First provisional results, each policy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 55 / 93 | 2 | 2 | 2 | 0 |
| 2 | 106 / 182 | 3 | 3 | 3 | 1 |
| 4 | 184 / 331 | 3 | 4 | 4 | 1 |

At the four-read cap, online rescue P95 on the 55 requests with
eligible windows is 67.77 ms for round robin, 61.08 ms for static,
and 61.09 ms for adaptive. Adaptive changes the selected order or
windows on 22 requests versus static, with identical accepted
evidence. The four distinct promoted views are in a private,
method-blind packet with two blank reviewer copies.

The full-pool audit uses all 251 original exact-window decisions
from the prior exhaustive GPU run (job 43026); every live decision
agrees with its reference score at the binary decision level.
Only four windows pass all criteria, across four requests. Only
one of those requests lacked an initial provisional recommendation.
Consequently the static four-read policy already reaches the
complete pool's provisional upper bound for those outcomes.
This ceiling is conditional on the frozen candidates, evidence
windows, and verifier. It is not a bound on real expert suitability
or another candidate and evidence pool. Coverage of eligible
personal-role evidence is the first bottleneck to change here.

[GPU aggregate](results.json) · [Full-pool audit](headroom.json)

## Reproduce

Use the authorized private development dataset and the prepared
panel and frozen trace described in the preceding benchmark. Keep
the replay trace and reviewer artifacts restricted. Run model
inference on a Slurm CUDA compute node and prepare the packet
locally.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_gap_budget_replay run \
    PRIVATE_FROZEN_PANEL PRIVATE_ROUTED_TRACE RERANKER_CHECKPOINT NEW_RESULTS.json PRIVATE_NEW_TRACE
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_gap_budget_replay packet \
  PRIVATE_DATASET PRIVATE_FROZEN_PANEL PRIVATE_NEW_TRACE PRIVATE_PACKET
```
