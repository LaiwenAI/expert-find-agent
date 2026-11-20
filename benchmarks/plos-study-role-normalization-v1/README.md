# Normalize paraphrased roles before per-report eligibility checks

Reuse the 59 source pairs and the pinned Qwen3-4B checkpoint from
[the clause selection study](../plos-study-clause-selection-v1).
The report identity and contributor initials are given. Define
30 new authored paraphrases of five publisher role fields plus
12 authored requests outside those fields. First measure role
mapping on all 42 phrases. The choice menu describes the scope of
the five available fields and an outside option. Reverse the menu
to test output encoding sensitivity. A two-call agreement policy
abstains whenever the predicted fields disagree.

For each matched report pair, use a deterministic hash to pick a
new phrase for each of its original two roles. The positive report
still assigns both roles to the person. The other report matches
exactly one role. Construct a second request that keeps the first
role and substitutes an outside-schema duty for the second.
Neither report provides an eligible pair for that request under
the five-role contract. It is treated as unsupported within this
schema, even when the paper might discuss the requested duty in
other sections.

Compare direct binary judgment for the whole task and report with
mapping each duty once and checking the known author's declared
fields deterministically. Both direct binary encodings see the same
new phrases and evidence. Repeated mapping results are cached in
the benchmark for the same phrase. Report the estimated uncached
two-role normalization time separately from local per-report
binding time. The benchmark excludes corpus lookup, startup,
queueing and throughput.

## Results

Both Slurm tasks in array 44151 completed on an RTX 5090 and
matched all counts. The fixture and checkpoint match the earlier
clause selection study. The covered-request analysis has 59 supported
and 59 unsupported report views. The outside-schema requests each
inspect both reports, for 118 additional negative views.

| Route | Supported accepted / 59 | In-schema wrong / 59 | Outside-schema wrong / 118 | Correct pairs / 59 |
| --- | ---: | ---: | ---: | ---: |
| Direct, 1 means yes | 59 | 48 | 85 | 11 |
| Direct, 0 means yes | 15 | 7 | 11 | 11 |
| Normalize once per role, forward menu | 59 | 0 | 20 | 59 |
| Normalize once per role, reverse menu | 55 | 0 | 9 | 55 |
| Use a normalized role only on agreement | 54 | 0 | 0 | 54 |

The forward menu maps all 30 covered phrases and abstains on
9/12 outside phrases. Reversing it maps 29/30 and abstains on
9/12. Requiring agreement maps 29/30 and abstains on all 12
outside phrases. The single covered abstention appears in five
positive report pairs. The repeated views share authored phrases
and should not be treated as independent tests of open-set risk.

At P95 the direct model takes about 15.7 ms per report. Adding the
two recorded per-duty times gives about 28.8 ms at P95 for one
menu encoding, or 57.3–57.5 ms for agreement. Per-report binding
after normalization takes about 0.02 ms. These are component
timings. The benchmark offers evidence for splitting reusable
query interpretation from per-report checking within a known
schema. Free prose, unseen publisher fields and full search
remain outside this evaluation.

[Both GPU runs](results.json)

## Reproduce

Reconstruct the pinned fixture using the
[binary encoding benchmark](../plos-study-label-order-v1).
Run in a Slurm GPU allocation using the pinned Qwen3-4B checkpoint.

```sh
export PYTHONPATH=src:.
python -m experiments.credit.study_role_normalization PINNED_FIXTURE NEW_RESULTS
```
