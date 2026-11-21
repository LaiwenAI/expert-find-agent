# Check the agreement proof before calling the paired role reader

The [prospective eLife replay](../paired-role-prospective-service-v2)
found higher proof coverage with reserved batched role agreement.
The extra role forwards caused a backlog. Freeze the same cohort,
search path, reserve policy and checkpoints. Add one exact
precondition to the paired fallback.

The reserve policy issues a two-role answer only if the role reader
agrees with the dense top choice for both required clauses. At
its entry, the original role sets have produced no proof. If the
two dense choices have no common contributor on the selected
sources, then agreement cannot issue a proof. Disagreement also
abstains. In those cases, skip the paired model forward and return
that same abstention. The test reuses the existing source selection
and same-owner lookup. It respects any prior source call and leaves
the preceding single-role route intact.

This argument assumes fixed sources and recorded role declarations,
and the deterministic agreement policy used here. Compare the
actual emitted paper, contributor and role tuple on every request,
with no credit for comparing only counts. Forward and token
telemetry may differ.

Use the same eLife loopback traces, 64 connections and one GPU
worker. Reload all three policies with the same models and scheduled
arrivals. Compare the baseline, original reserve agreement and
prescreened reserve agreement at both rates in two execution orders.
Report distinct-request proof counts, client and server tail,
completed throughput, role forwards and nonpadding role-prompt
tokens. The input conditions are supplied fragments and do not
include general free-text parsing.

## Results

Slurm array 46162 completed both 48-second tasks on ws3 RTX 5090.
Every one of the 108 distinct responses has the same evidence tuple
under original and prescreened agreement. Their response signatures
also match across both request orders and rates. The original
reserve and baseline signatures reproduce the earlier benchmark.

| Actual arrival rate | Route | Client P95 (ms) | Server queue P95 (ms) | Achieved req/s | Role forwards | Role tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 29.0 | Single-slot baseline | 52.8–52.9 | 35.6–36.7 | 29.0 | 24 | 2314 |
| 29.0 | Reserved batched agreement | 78.2–85.5 | 53.5–62.7 | 28.9 | 118 | 16502 |
| 29.0 | Prescreened batched agreement | 56.5–58.5 | 36.6–38.1 | 29.0 | 44 | 5334 |
| 62.1 | Single-slot baseline | 99.0–106.5 | 83.8–91.8 | 61.0–61.2 | 24 | 2314 |
| 62.1 | Reserved batched agreement | 1227.8–1230.0 | 1203.5–1205.1 | 45.4–45.5 | 118 | 16502 |
| 62.1 | Prescreened batched agreement | 196.8–215.3 | 175.4–195.5 | 58.7–59.0 | 44 | 5334 |

Forward and token totals refer to each 216-request replay. The
prescreen removes 74 of 118 role forwards (62.7%) and 11,168 of
16,502 role prompt tokens (67.7%), preserving all 30 distinct
strict proofs and zero observed adverse issuances. The single-slot
baseline returns 22 strict proofs. Both agreement policies return
12/27 keyword and 18/27 excerpt positives.

At the higher arrival rate most of the reduction is in queueing.
The optimized reserve route remains slower than the baseline while
covering more positives. These experiments reuse the same structured
eLife requests and one resident worker. They establish output
preservation for this routing rule and workload, with quality
and capacity for complete enterprise requests still to validate.
[Full results](results.json)

## Reproduce

Use the exact artifacts and command from the
[prospective service comparison](../paired-role-prospective-service-v2#reproduce).
Add `--prescreen` and repeat with `--reverse`, using a fresh output
for each Slurm run. This selects the three policies above.
