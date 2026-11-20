# Bind the person, project, role and hard constraint to one measurement

This diagnostic extends the [transfer constraint counterfactuals](../transfer-constraint-counterfactual-v1)
with fictional contributor names and structured project provenance.
It isolates a publication claim: can an otherwise valid empirical
measurement be used only for the person, project and role that it
actually supports?

The 480 cases use 48 base scenarios from 12 domains. Each base scenario
has one qualifying bound and one rejecting bound over the same actual
measurement. For each bound:

- **Intact** retains empirical evidence for the named test lead,
  project, metric and operating setting.
- **Wrong owner** changes the contributor on the valid record.
- **Wrong project** changes its project while retaining the person.
- **Wrong role** changes test lead to report editor.
- **Planned only** replaces the empirical statement with a target.

Each batch of four shuffled cards includes three distractors. They
supply a different operating setting, a proposal in the requested
setting, and another person's measurement in another project.
The missing conjunction cases are unknown based on the given evidence.
This does not say the named person lacks the capability.

Compare a frozen Qwen3 4B decision under two digit-to-label orders with a deterministic gate that
checks role, owner, project, metric, setting and measurement type
before comparing the raw-text number and units. Provide both
approaches the request, target owner and project, metric and setting,
and the same rendered cards. The single-constraint ablations remove
only the owner or project check. Ground-truth intervention labels
and normalized values are excluded from model inputs.

The structured checks intentionally target the authored schema.
They test constraint composition, with no claim of performance on
arbitrary text, open candidate discovery, human attribution or
unseen eligibility settings. Do not split individual variants of
a base scenario across development and test.

## Results

Both Slurm tasks in job `44102` on ws5 reproduce all non-timing counts.

| Route | Correct / 480 | Both intact bounds correct / 48 | False eligibility on unavailable evidence / 384 |
| --- | ---: | ---: | ---: |
| Frozen 4B, original order | 349 | 5 | 86 |
| Frozen 4B, rotated order | 64 | 8 | 130 |
| Structured, omit owner | 384 | 48 | 48 |
| Structured, omit project | 384 | 48 | 48 |
| Structured, bind owner and project | 480 | 48 | 0 |

The two structured ablations solve every intact numerical pair.
Removing the owner check nevertheless accepts 48 wrong-owner views
and rejects the other 48 without valid evidence for the specified
person. Removing the project check has the same pattern on
wrong-project views. These comparisons isolate two different
attribution errors that would be invisible on numeric pairs alone.

The 4B decision changes materially when the numeric label order is
rotated. Report both mappings. The apparent overall accuracy of the
original order is influenced by the 384 unknown cases. Neither
mapping reliably solves both intact bounds. Measured decision
P95 is about 24.6 ms for each 4B mapping and 0.026 ms for the joint
CPU gate, with the latter receiving explicit role records and
target fields. These timings exclude the upstream work needed to
construct equivalent structured records from authentic documents.

[Dataset](cases.json) · [Aggregate results](results.json)

## Reproduce

```sh
PYTHONPATH=src:. python -m experiments.transfer.owner_scope_counterfactual build NEW_CASES
PYTHONPATH=src:. python -m experiments.transfer.owner_scope_counterfactual run NEW_CASES NEW_RESULTS
```

Run the decision comparison on a Slurm CUDA allocation with the
pinned Qwen3-4B-Instruct checkpoint from the original transfer study.
The structured route runs on the host CPU. Report latency as local
decision time, without source discovery or network.
