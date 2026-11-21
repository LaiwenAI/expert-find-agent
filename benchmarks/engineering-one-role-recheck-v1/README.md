# Recheck one ambiguous role against a two-source proof

Use the [engineering cross-source panel](../engineering-cross-source-search-v1)
with the two designated sources supplied. The 30 held requests
include 15 positives and 15 role-to-source swaps over four article
pairs. Each clause starts as the previous ridge role band. A
contributor must support every retained role on the corresponding
paper. Keep any initial proof unchanged. On unresolved requests,
a second pass can narrow at most one band to one role. The other
clause and both sources remain fixed.

Compare the following selectors at the same maximum budget.

- **Left or right priority:** recheck the corresponding ambiguous
  clause first, falling back to the other if only one is ambiguous.
- **Smallest margin:** select the ambiguous clause with the smallest
  dense gap between its two leading roles.
- **Proof-aware:** enumerate the branch for each possible answer
  inside each band. Select the clause with the highest fraction of
  answers that could yield a same-person proof. Use the dense
  margin to break ties.
- **Proof-aware with skip:** apply the same choice, and avoid the
  model call if every branch remains unsupported.

Use Qwen3-4B-Instruct-2507 with only the phrase and role definitions
from the selected band. Include a none-of-these option. The reader
does not see authors or which roles would yield a proof. Choose a
margin cutoff for the proof-aware route on the previous 48 scripted
[PLOS calibration cases](../engineering-joint-calibration-v1) over
three paper pairs. The cutoff maximizes calibration proofs subject
to a 5% adverse case cap. Use the same cutoff for every target arm.
The role reader processes the request phrase, without reading
additional article text.

In the oracle replay the selected role is corrected only if the
intended role is in the original band. The best-single oracle can
also choose which clause to fix using the label. Both oracles
keep the other band intact. These runs isolate the value of
selection from the value of accurate single-clause interpretation.

## Results

| Route | Strict proofs / 15 | Adverse / 30 | Model calls / 30 | Prompt tokens total | Mean role-stage ms | Role-stage P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge band | 1 | 0 | 0 | 0 | 9.5–9.6 | 9.7–9.9 |
| Left priority | 6 | 0 | 29 | 7,903 | 27.6 | 35.7 |
| Right priority | 4 | 0 | 29 | 7,903 | 27.6 | 35.7 |
| Smallest margin | 5 | 0 | 29 | 8,821 | 29.1–29.2 | 35.7 |
| Proof-aware | 6 | 0 | 29 | 8,855 | 29.3 | 35.7 |
| Proof-aware with skip | 6 | 0 | 10 | 3,321 | 16.8 | 35.6 |

On PLOS, the fitted cutoff is zero, with 24 strict positives and
zero adverse cases on the calibration data. With correct single
role answers, left priority and proof-aware selection each reach
9 proofs, right priority reaches 7, and smallest margin reaches 8.
The best-single oracle also reaches 9. Fixing both roles with
their labels provides all 15 by the original panel construction.

Proof-aware selection ties left priority at 6/15, so these
four source pairs do not establish a quality benefit from the
adaptive choice. The skip removes 19 of 29 otherwise attempted
calls and preserves every output. This follows from the branch
rule: if no role in either band can complete the proof with
the other band intact, a confined recheck cannot issue a result.
The reduction in calls and mean latency does not translate into
a material P95 change. A skipped call says nothing about
whether another source or a role outside the bands can provide
evidence. The prototype-band reference previously found 4/15
with a roughly 10 ms role-stage P95
([baseline](../engineering-role-comparison-v1)).

Both tasks of Slurm array 45924 on ws3 RTX 5090 reproduce all
non-timing fields. Latencies are factored resident-model estimates.
Each route is charged for initial role encoding, branching and
its selected live read only. The harness times both candidate reads
separately for paired comparison. Source discovery, unrestricted
query parsing and network transport are outside this timer.
The 30 cases reuse four paper pairs. Zero observed adverse
results is a fixture count; field risk remains to be measured.
[Full results](results.json)

## Reproduce

Restore the engineering panel and PLOS projection from the linked
benchmarks. Use the frozen embedding and reader revisions recorded
in the results. Run in Slurm CUDA with a fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.engineering_one_role_recheck \
  ENGINEERING_PANEL PLOS_PROJECTION NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```

Repeat with `--reverse`.
