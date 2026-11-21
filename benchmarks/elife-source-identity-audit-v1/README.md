# Diagnose source identification before the final role check

Use the [expanded eLife holdout](../elife-extended-role-holdout-v1)
as a diagnostic panel after its first evaluation. Replay the archived
lightweight cascade and calibrated source gate. Require their exact
output signatures with both papers in the catalog and with either
paper withheld. Freeze the original gate, topic rankings, candidate
pair and role results for this test.

Inspect only the source pairs passed to the calibrated gate's role
reader. Test whether cheap checks could block a fallback before that
forward is spent.

| Check | Required on both candidates |
| --- | --- |
| Rank cap | Each paper appears within its own clue's fused top 3, 6, or 12 |
| Own-clue match | Each clue has higher dense similarity to its proposed paper than to the other paper |
| Keyword coverage | The indexed text contains at least 15%, 30%, or 45% of the observed clue's IDF-weighted unique words |
| Independent source leader | Each proposed paper has the best fitted resident source score among its clue's fused top 24, regardless of author or role |
| Leader or top three | Independent source leaders, or both papers in the fused top three |
| Combined corroboration | Own-clue match and at least 30% weighted coverage |

Count retained exact and inexact source-pair proposals independently
of the role decision. Reconstruct what each guard would issue by
keeping the archived fallback response when it passes and returning
the cascade's response otherwise. Report exact proofs, adverse
issuances, and substitute issuances when an intended paper is
absent. The quality results are causal for this threshold-only
replay because the checks do not change the underlying search,
pair choice or role scoring. Actual latency of the candidate checks
is outside this experiment. No held threshold selection or
independent confirmation is claimed. Withholding is repeated over
the same underlying paper and author groups.

## Diagnostic results

Both tasks in Slurm array 46439 reproduce the frozen response
signatures and all aggregate outcomes on ws3 RTX 5090. The gate
passes nine keyword proposals to the role reader in the intact
catalog, with five exact source pairs and four substitutes.
It passes 15 substitutes when one intended paper is withheld.
It produces no distinct excerpt fallback.

| Policy | Intact keyword strict / adverse | Keyword substitute issuances after withholding |
| --- | ---: | ---: |
| Lightweight cascade | 41 / 1 | 1 |
| Calibrated fallback | 46 / 4 | 16 |
| Fused top three on both sides | 43 / 2 | 1 |
| Fused top six on both sides | 44 / 4 | 10 |
| Own-clue cosine | 44 / 1 | 6 |
| At least 30% IDF coverage on both sides | 42 / 1 | 1 |
| Both individually lead by the resident score | 42 / 1 | 1 |

The direction-only check removes every inexact complete-catalog
proposal, yet it retains five keyword substitutes under removal.
The strongest individual source check and the coverage check
match the cascade on observed adverse issuance but retain one
of the five added proofs. The fused top-three check retains
two proofs and leaves one new adverse issuance. These patterns
motivate a live check of requiring both top-three membership and
the own-clue cosine comparison. That conjunction was selected
after looking at this diagnostic.

## Live combination on the same panel

Run both checks before the 4B role reader. Require fused top-three
membership for both proposed papers and a strictly larger cosine
similarity for each clue against its own paper than against the
other proposed paper. The implementation reuses the rankings and
dot products already computed by the search. It adds no model call
for the guard itself.

Both tasks in Slurm array 46446 reproduce the previous cascade and
ungated fallback signatures. The guarded arm also reproduces its own
outputs in both orders.

| Live policy | Strict keyword proofs / 72 | Intact adverse / 144 | Missing-source substitute issuance / 288 | Verifier forwards with intact catalog | Intact keyword serial P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lightweight cascade | 41 | 1 | 1 | 34 | 26.97–27.37 ms |
| Calibrated fallback | 46 | 4 | 16 | 43 | 28.35–28.45 ms |
| Top three and own-clue cosine | 43 | 1 | 1 | 36 | 27.71–27.75 ms |

The guarded arm uses two additional 4B role forwards relative to
the cascade and retains 53 exact excerpt proofs. There are no new
adverse issuances under either availability condition. The total
includes the baseline adverse cases. The paired descriptive interval
for the added strict keyword proofs is 0 to +6.94 percentage
points across connected paper and recorded-author groups. The JSON
uses the bootstrap helper's general sampling-unit label. This run
tests implementation and replay stability on the same data used to
select the conjunction. Independent accuracy and risk both remain
unverified. Client queueing is outside the serial timings.

[Full diagnostic and live replay results](results.json)

## Reproduce

Use the [complete-catalog and source-withholding benchmark](../elife-extended-role-holdout-v1#reproduce).
Add `--source-audit PRIOR_RUN_JSON`, where `PRIOR_RUN_JSON`
is either individual prior output. The corresponding object can
also be extracted from the published aggregate `runs` array.
Repeat with `--reverse` and a fresh path. For the live conjunction,
replace `--source-audit PRIOR_RUN_JSON` with `--guarded-source-replay`.
