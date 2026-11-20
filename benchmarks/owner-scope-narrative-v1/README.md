# Recover jointly bound measurements from prose

This diagnostic renders the 480
[matched owner and project cases](../owner-scope-counterfactual-v1)
as 960 English prose requests. The plain style rewrites four
provenance fields into a sentence and its measurement statement.
The distractor style adds the other person as an observer and an
additional project mention for the contributor. The target fact,
outcome label and measurement remain unchanged across renderings.

Compare the frozen 4B direct decision under the two original label
orders with a factorized route. The factorized route asks the same
model to select one supporting card or return none without seeing
the requested numerical threshold. A deterministic comparison then
parses the visible bound, quantity and units from raw text. A
second route requires another 4B judgment on the selected passage
before the same arithmetic. The gold-card control identifies the
best possible result of this arithmetic with correct evidence.
The model sees the target person, project, metric and setting but
does not see the source labels or normalized quantities.

Measure selected-card correctness and report false eligibility on
requests with no valid evidence. Report both-intact-pair accuracy
rather than relying on overall accuracy of the majority-unknown
panel. Time factorized decisions as if source selection and the
optional check were recomputed for every request. Physically reuse
a result only on identical selector inputs, including card order,
and charge the original forward time to every matching request.

These are authored, matched evidence rooms. They stress attribution
in prose under controlled confounders. The results apply to the
frozen 4B checkpoint and prompts, with no independent enterprise
evaluation or authentic document field-extraction claim.

## Results

Both Slurm tasks in job `44111` on ws5 reproduce all decision and
selection counts. Timing below comes from the first run.

Each style contains 96 intact requests and 384 requests without
matching evidence. The two figures in each cell are plain / distractor
style.

| Route | Both intact thresholds correct / 48 | False eligibility without matching evidence / 384 | Cold-path P95 |
| --- | ---: | ---: | ---: |
| Frozen 4B direct | 5 / 0 | 235 / 252 | 24.67 / 28.83 ms |
| Frozen 4B direct, rotated labels | 8 / 3 | 264 / 320 | 24.64 / 28.85 ms |
| Select, then compare | 36 / 29 | 182 / 165 | 21.21 / 24.85 ms |
| Select, verify, then compare | 22 / 13 | 121 / 111 | 35.85 / 39.66 ms |
| Compare with gold card | 48 / 48 | 0 / 0 | 0.03 / 0.03 ms |

The selector picks the correct card on 52 / 96 plain intact requests
and 31 / 96 with distractors. It returns none on only 3 / 384 and
10 / 384 requests that lack matching evidence. Another 4B check
reduces false eligibility and still leaves substantial unsupported
decisions, adds latency, and reduces correct paired decisions.

The threshold score can mask attribution errors. In the plain
condition, 36 successful pairs account for 72 intact decisions.
At most 52 of those decisions could have used the correct card.
Therefore at least 20 got the answer with the wrong evidence.
The corresponding lower bound is 27 decisions with distractors.
Even an accurate comparator cannot recover the intended evidence
after a wrong source is selected. The gold-card control verifies
that the arithmetic itself can handle the authored thresholds.

[Aggregate results](results.json)

## Reproduce

The optional build command reconstructs the prose views without
duplicating the parent dataset in the repository. The run command
regenerates the same cases and uses the pinned local 4B model on
a Slurm allocation.

```sh
PYTHONPATH=src:. python -m experiments.transfer.owner_scope_narrative build NEW_CASES_JSONL
PYTHONPATH=src:. python -m experiments.transfer.owner_scope_narrative run NEW_RESULTS
```
