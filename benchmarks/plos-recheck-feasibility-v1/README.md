# Skip structurally impossible role rechecks

The [frozen service comparison](../plos-frozen-role-recheck-service-v1)
found one additional correct keyword proof and a substantial tail
latency increase. This replay keeps its source proposals, source
cutoff, role model, 4B prompt, and issuance threshold. It asks how
many 4B calls can be eliminated before reaching the model.

For a candidate pair of two papers, intersect their recorded
contributors. Each non-rechecked role remains fixed to the original
query label. A rechecked role can be any declared role on that side.
If no common contributor can satisfy both sides, the existing route
cannot issue a proof for that pair for any 4B output. Skip that call.
This condition applies to the frozen selected sources and known
declarations. A route that can subsequently look for further sources
or resolve missing author declarations would have a different
decision space.

Also test an approximate filter that limits the feasibility check to
the 0.6B model's first two labels for each uncertain role. For all
requests that pass either filter, the 4B model still sees the full
taxonomy, and the original proof check remains unchanged. This makes
the top-two restriction a choice about whether to spend a call, with
no effect on how admitted calls decode their roles.

Measure both filters through the same single-worker loopback service
as the unfiltered 4B policy and the original lightweight baseline.
Require complete proof equality between exact filtering and unfiltered
4B on every request. The baseline's already-issued proofs remain
unchanged under all policies. After both runs, match each filter's
realized call count to an offline score-only allocation, selecting
the unfiltered eligible requests nearest to the ridge threshold.
Resolve ties from request fields. This same-work control observes
the realized budget for each availability group and does not
represent an online latency comparison.

The 13 groups have already been evaluated. No threshold is fitted
here. The experiment estimates replay work and latency, not the
risk or coverage of a new population.

## Results

Both RTX 4090 tasks in Slurm array 47368 reproduced the archived
baseline and unfiltered 4B decision counts. Exact feasibility
preserved all proofs and abstentions across 234 requests. The
top-two filter also preserved them on this panel.

| Availability and route | Correct / wrong | 4B calls | 4B tokens | Client P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Intact, lightweight | 9 / 0 | 0 | 0 | 17.1–17.3 |
| Intact, unfiltered 4B | 10 / 0 | 47 | 24,496 | 79.8–80.1 |
| Intact, exact feasibility | 10 / 0 | 17 | 9,436 | 78.8–78.9 |
| Intact, top-two feasibility | 10 / 0 | 13 | 7,295 | 56.3 |
| Withdrawn, lightweight | 0 / 1 | 0 | 0 | 17.0–17.2 |
| Withdrawn, unfiltered 4B | 0 / 1 | 54 | 27,522 | 75.8–78.7 |
| Withdrawn, exact feasibility | 0 / 1 | 0 | 0 | 17.2–17.5 |
| Withdrawn, top-two feasibility | 0 / 1 | 0 | 0 | 17.3–17.5 |

With exact filtering, 84 of 101 recheck calls and 42,582 of
52,018 input tokens are removed. On the intact workload,
enough requests still run a full forward to dominate P95.
The offline equal-call confidence control also keeps the one
extra intact proof at both budgets. The replay therefore
establishes output-preserving work reduction for this
fixed pair and complete declarations. The approximate
filter has a promising latency result with too few recovered
proofs to separate it from the score control. All routes
retain the baseline's one wrong proof on withdrawal.

[Full runs and input digests](results.json)

## Reproduce

Use the [frozen service arguments](../plos-frozen-role-recheck-service-v1#reproduce),
substituting `experiments.credit.plos_recheck_feasibility_service`.
Use a new output path and repeat with `--reverse`.
