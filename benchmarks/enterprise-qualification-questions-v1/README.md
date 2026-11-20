# Ask only about qualifications that may change the search result

Reuse the [qualification replay](../enterprise-qualification-replay-v1)
on the same 25 enterprise development requests. Fix people,
evidence, verified scores, and original order. A condition accepted
by both parsers remains required. Every disputed clause is a binary
unknown about whether the requirement applies. The original request
interpretations are not judgments of which answer is correct.

For each person with topic support and all agreed qualifications,
record which disputed constraints their inspected evidence fails.
Under a given answer assignment, the person passes if none of those
failed constraints is active. Keep the original ranking order.
Enumerate the binary assignments and evaluate two clarification goals:

- Whether any candidate can receive a provisional recommendation.
- The exact set of provisional candidates, with order held fixed.

Choose the next clause by dynamic programming, minimizing the
worst-case number of questions, then the total number across all
assignments, then the original clause order. Stop once the outcome
is constant across the remaining assignments. Compare this decision
cost with asking about every disputed clause. The reported uniform
scenario is a calculation under equal assignment weights. It does
not estimate how users answer or the effect of asking in live traffic.

If a clause is unresolved, recommendations require evidence under
every retained interpretation. The empty decision set is distinct
from confirmation that no suitable expert exists outside the fixed
shortlist. This is an offline decision analysis with no simulated
human labels, no live user interaction, and no claim about the
accuracy of the frozen model margins. The panel has at most four
disputed clauses per request; exhaustive enumeration here is not a
latency claim for unconstrained request lengths.

## Results

Both tasks of Slurm array `44893` on `ws3` reproduce the
published fixed-evidence replay and give identical decision counts.

| Diagnostic on the 25 requests | Count |
| --- | ---: |
| Requests with parser disagreement | 21 |
| Disputed clauses across those requests | 43 |
| Request outcomes invariant under every modeled interpretation | 18 |
| Requests in which provisional eligibility can change | 7 |
| Requests always resolved by one targeted question | 4 |
| Total questions across requests, summing each request's worst case | 13 |

For these requests the exact candidate-set goal gives the same
question count as the availability goal. In each of the seven
sensitive requests, at least one answer to the first question is
enough to resolve availability. Across all 25 requests, the sum of
the uniform-assignment expected question counts is 9.125.
Under all modeled interpretations jointly, no request has a
guaranteed provisional recommendation from the inspected shortlist.
This is a bound conditional on the supplied evidence and margins.

The comparison identifies 14 requests with parser disagreement
but invariant output. Asking about those disagreements cannot alter
the frozen result. This design separates deciding whether to ask
from deciding whether any original eligibility requirement applies.

[Aggregate results](results.json).

## Reproduce

Use the pinned replay inputs and run on a Slurm GPU with a fresh
output file. Use the original verifier and add the flag below.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_qualification_replay \
  DEVELOPMENT_QUERIES PRIVATE_CORPUS FIRST_ROUND_TRACE FIRST_ROUND_AGGREGATE \
  PRIVATE_SCOPE_TRACE SCOPE_AGGREGATE QWEN_RERANKER NEW_AGGREGATE --clarification
```
