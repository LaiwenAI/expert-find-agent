# Measure the one-clause proof ceiling before changing the reader

Reuse the [development](../elife-one-role-transfer-v1) and
[held](../elife-prompt-holdout-v1) eLife cohorts with both target
sources available. Keep source retrieval, dense role banks,
prototype agreement and the name-only reader fixed. This study
dissects loss of strict positive proofs and tests whether
choosing a slot after computing possible approved proofs helps.

For each correctly retrieved positive, replay the following
diagnostic interventions using labeled roles. These labels never
enter the executable decision paths.

1. Keep the initial strict proof, if present.
2. Give the selected slot its correct role when it appears in
   both the original approved branch and the two-option shortlist.
3. Remove the two-option cap while keeping the selected slot
   and its original approved branch.
4. Permit choosing either originally approved slot.
5. Permit the correct role outside the approved branch for
   one slot, holding the other role set fixed.
6. Correct both roles while retaining the selected source pair.

Each intervention must still produce the exact contributor set.
When the source pair is wrong, none of these role-only
interventions can produce a strict proof. The ceilings describe
available improvements with exact answers, not model performance.

Test two executable alternatives at the same one-reader-call
limit. Both check prototype membership and proof feasibility
for every unresolved clause first. Choose the clause with the
highest fraction of approved roles, or the highest approved
softmax mass over the mean standardized dense scores. Break ties
by the original dense gap. Then use the same name-only reader,
role cutoff and final proof check. Neither alternative is tuned
on the held cohort.

## Results

| Cohort and clue | Positive requests | Correct sources | Initial ridge proof | Correct choice in two options | Correct choice in selected slot | Correct choice in either slot | One role with expanded options | Both roles corrected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development excerpt | 51 | 51 | 13 | 30 | 31 | 31 | 38 | 51 |
| Development keywords | 51 | 36 | 9 | 22 | 23 | 23 | 28 | 36 |
| Held excerpt | 66 | 66 | 22 | 44 | 45 | 45 | 55 | 66 |
| Held keywords | 66 | 48 | 18 | 34 | 34 | 34 | 40 | 48 |

In the held excerpt view, the name-only recheck produces 42 exact
proofs, with six adverse outcomes among 48 issued results. The
current selected slot has a ceiling of 45. Selecting either slot
does not increase it. Correcting one role outside its band or
agreement guard raises the ceiling to 55. Correcting both reaches
66. Thus, at the current source exposure, three positive proofs
are available through perfect selection inside the live decision
space, ten more through expanding its one-role action, and eleven
more require changing both roles relative to the initial bands.
The keyword view has an additional 18 unavailable source pairs.

The development excerpt recheck has 29 strict proofs and one
adverse outcome. The corresponding held keyword recheck has 33
strict proofs and one adverse outcome. Neither the fraction-based
choice nor the score-weighted choice changes a selected reader
slot or output in either cohort. Their timed paths retain about
26.4–26.6 ms held excerpt P95 and 21.6–22.0 ms keyword P95.
The timing shares the small two-slot approval calculation and
charges each path for its one selected live reader. These results
prioritize testing a bounded expansion of role alternatives over
more complex slot ranking. An expansion needs its own wrong-role
risk guard. The oracle increments measure recovery opportunity,
not precision after deploying that expansion.

Both tasks in Slurm array 45992 on ws3 RTX 5090 reproduce all
non-timing fields with reversed order. The local timing excludes
unstructured request parsing and HTTP transport. The data reuse
13 held article pairs and the fixed NISO wording bank. [Full results](results.json)

## Reproduce

Restore both pinned eLife projections using the published source
manifests. Run on an allocated Slurm CUDA GPU and repeat with
`--reverse` and a fresh result path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.role_recheck_opportunity \
  DEVELOPMENT_PROJECTION HELD_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```
