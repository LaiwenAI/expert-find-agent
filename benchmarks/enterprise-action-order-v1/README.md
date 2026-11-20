# Decide when to clarify a qualification and when to expand evidence

Replay the 25 frozen multi-condition development requests on which
the original system withheld a recommendation. They are the same
requests, people, excerpts, and interpretation alternatives as in the
[scope and evidence-depth study](../enterprise-qualification-scope-depth-v1).
The initial search returns five people with two passages each. An
expansion can read two passages for the next 15 people, read up to eight
passages for the existing five, or run both expansions as a reference.
The frozen 0.6B verifier scores each topic or qualification against the
planned excerpts. An unseen expansion and its scores are hidden from
the policy until it chooses that action.

A qualification accepted by both parsers is mandatory. For each
disputed clause, enumerate both possibilities. A clarification reveals
one disputed bit in the current hypothetical scenario. A policy asks
at most one question, and only when at least one answer would leave a
model-supported candidate across all interpretations still possible.
Among such questions it favors the greatest number of eligible
assignments, then the earlier clause. Later rule checks intersect the
candidate sets for all surviving interpretations. The fixed read plans
are nested, so an expansion reuses excerpts already inspected.

The state-dependent policy first attempts a question on the shallow
result. With no safe candidate, it expands people if topic support is
absent or at least one candidate can pass under a retained interpretation.
Otherwise, it expands passages for the existing five. If it has not
already asked a question, it may ask after this expansion. The fixed
policies compare both orderings for each expansion, plus controls that
use only reading or only clarification.

Each question receives the stipulated answer for that scenario. The
reported mean weights interpretations equally within each request and
then sums across requests. It is a sensitivity calculation. Approved
candidates are provisional model outputs. Eventual user answers,
independent eligibility and latency need separate measurements.

## Results

Both tasks in Slurm array 45615 on a ws3 RTX 5090 replay 25 requests and
122 possible interpretations. Four requests admit a useful shallow
question. The table sums within-request, equally weighted means. The
last column counts requests for which at least one hypothetical answer
sequence leads to a model-supported result.

| Policy | Checked excerpts | Questions | Requests supported, scenario-weighted sum | Requests supported in any scenario |
| --- | ---: | ---: | ---: | ---: |
| Shallow only | 250 | 0 | 0 | 0 |
| Shallow then ask | 250 | 4 | 2 | 4 |
| Widen people only | 997 | 0 | 2 | 2 |
| Deepen passages only | 869 | 0 | 1 | 1 |
| Widen then ask | 997 | 6 | 5 | 8 |
| Ask then widen | 937 | 7 | 5 | 8 |
| Deepen then ask | 869 | 4 | 3 | 5 |
| Ask then deepen | 824 | 5 | 3 | 5 |
| Route based on initial evidence | 904 | 7 | 4 | 7 |
| Both expansions then ask | 3,344 | 7 | 5.5 | 9 |

Asking before widening saves 60 inspected excerpts and adds one
clarification in the scenario mean, with the same aggregate count of
supported requests. Asking before deepening saves 45 and adds one.
Worst-case excerpt counts are the same for both orderings of each
fixed expansion. The state-dependent rule offers no joint improvement
over the fixed ask-then-widen policy in this panel. Compared with that
policy, reading both expansions uses 2,407 more excerpts for 0.5 more
model-supported requests in the sensitivity calculation.

The results bound the benefit of the actions under these two model
interpretations and the fixed evidence pools. Compare readings and
clarifications in their original units. Service cost needs measured
latency, caller effort and evidence adjudication. The
[SIGIR clarification task](https://doi.org/10.1145/3331184.3331265)
provides a reference for studying user answers and question quality.

## Reproduce

Use the frozen private inputs from the
[scope and evidence-depth study](../enterprise-qualification-scope-depth-v1).
Add `--action-audit` to its invocation on a Slurm GPU. Repeat with
`--order-offset 1`. The researcher retains the private trace inside the
authorized workspace. The code and aggregate replay output are published
here.
