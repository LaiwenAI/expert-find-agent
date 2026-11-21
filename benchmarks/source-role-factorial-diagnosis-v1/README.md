# Separate source nomination and role verification under supplied spans

The [wording-transfer study](../transfer-source-role-resolution-v1)
found low strict proof coverage even when the search received the
reference topic and activity fragments. Freeze that input interface
and the existing verifier. Run one positive and one role-swap
control per article pair across the current eLife holdout, earlier
eLife and engineering. Query with the original keywords and the
unindexed half of each abstract. For each view, replay both sources
available, left source hidden and right source hidden.

## Diagnostic interventions and deployable routes

Check that the instrumented baseline produces the same response
as the existing HTTP service handler on every fully visible case.
Separate source exposure in the fused top six and top 24 from exact
source selection and strict owner–role proof. Label the first failing
stage on positive requests under each view. The labels follow the
baseline route and do not assume that later stages would pass.

- **Supply roles only.** Fix the selected pair, replace the detected
  role sets with the intended labels, and test the same-owner join.
- **Supply sources only.** Fix the pair to the reference papers and
  run the same role route. In the budget-matched arm, keep the role
  verifier unavailable if the observed route had consumed the one
  verification call on source evidence. Report a separate arm where
  the corrected source frees a possible role call.
- **Supply both.** Check that the declared same-owner join agrees
  with the benchmark labels.

Compare the existing screened one-call route with two deployable
alternatives. The scan-four route checks role feasibility for the
first four distinct linked proposals, passing at most one to source
verification. The deeper anchor route extends the original
two-sided cosine requirement to linked pairs from the top 24. It
can then spend its single call on roles. It otherwise uses the
original source backup. Neither route sees target sources or role
labels. The held-source controls count wrong-source issuance after
the role checks.

## Results

Both Slurm tasks in array 46126 completed on ws3 RTX 5090 and
reproduced all non-timing counts in reverse request order. The
cohorts are the 393-paper current eLife catalog with 13 pairs, the
280-paper earlier eLife catalog with 10 pairs, and the 53-paper
engineering catalog with four pairs. All tables below use the
27 positives once per clue view.

| Clue view | Positives | Both sources top 6 | Both top 24 | Correct pair selected | Baseline strict | Role labels supplied | Sources supplied, matched budget | Sources supplied, one role call | Both supplied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Keywords | 27 | 23 | 24 | 21 | 15 | 21 | 17 | 17 | 27 |
| Excerpt | 27 | 27 | 27 | 27 | 17 | 27 | 17 | 17 | 27 |

For keyword clues, six of the twelve baseline misses occurred after
correct source selection. Replacing only the roles fixes those six.
Replacing only the sources at the matched role budget fixes two.
Four require both interventions. The two eLife transfer cohorts
and engineering retain distinct error mixes.

For the richer excerpt clues the correct pair was selected on all
27 positives. The baseline returned 17 strict results. Nine of the
ten misses came from extra plausible labels treated as joint
requirements, and one from a missing required label. Supplying
roles repairs all ten. Giving a corrected source an extra role call
when possible does not improve these aggregate counts.

Neither bounded expansion changed the aggregate outcomes or call
counts on any cohort, view or availability condition. There were
zero wrong-source issuances across the 216 missing-source cases
per deployable route. These are two hidden-source variants, two
clue views, and 54 base queries. Their repeated inputs and limited
domains provide little evidence about population risk.

Prioritize a one-call interpretation of both ambiguous clauses
while preserving the exact same-owner proof and separate controls
for role swaps and unavailable sources. The factorization supplies
benchmark labels only for diagnosis. The deployable timing measures
resident embedding, retrieval and verification, and excludes text
parsing, index build, HTTP and queueing. Plans are timed in a fixed
within-request order, so use the P95 as descriptive.
[Full results](results.json)

## Reproduce

Use the frozen projections, manifests and checkpoints from the
[transfer study](../transfer-source-role-resolution-v1#reproduce).
The CLI uses the same positional inputs. Within a Slurm CUDA
allocation, replace the module with
`experiments.credit.source_role_factorial_diagnosis`, remove
`--seed`, and run in both request orders with new output paths.
