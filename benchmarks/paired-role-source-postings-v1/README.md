# Recover sources using role-compatible author postings

Reuse the 108 structured eLife requests from the
[stage audit](../paired-role-stage-audit-v1). Preserve the stock source
selection and exact role agreement. Consider another source action
only when no pair was selected and the shared forward is still free.

Filter each clause's top 24 topical sources by the frozen static
prototype role band for that clause. Join the surviving postings
on a declared contributor, excluding identical papers. Rank the
distinct pairs by the smaller worst-side cosine deficit, then fused
rank sum and document IDs. Neither side needs to lead the topic ranking.
Only the first pair can use the remaining forward. Compare weak-side
confirmation (one verifier row) with both-side confirmation (two rows
in one forward), applying the old source cutoff to every scored row.
A recommendation still requires the same contributor to have the
declared roles on both papers. No test labels enter routing.

Count whether the target was feasible, ranked first, and verified.
Repeat with each target source individually withheld. Preserve the
request wording, query representation, and all model thresholds.
Assert exact response parity with the archived stock proof under
full availability and check the one-forward budget on every request.
These are post hoc component results on an already used public panel.
They measure model work and outcomes without timing the service.

## Results

Both tasks in Slurm array 46235 completed on ws3 RTX 5090.
Results and evidence signatures match in both request orders.
The stock results match the archived full-availability proof.

| Policy | Keyword strict / adverse | Excerpt strict / adverse | Full-availability source forwards / rows | Adverse when one target source is withheld |
| --- | ---: | ---: | ---: | ---: |
| Stock | 12 / 0 | 18 / 0 | 4 / 4 | 0 |
| Postings, verify weak side | 12 / 0 | 18 / 0 | 5 / 5 | 1 |
| Postings, verify both sides | 12 / 0 | 18 / 0 | 5 / 6 | 0 |

The removal conditions cover 216 request instances and share the
same source pairs across wording and role controls. One-side
confirmation issued a wrong-source substitute on one missing-left
excerpt instance. Both-side confirmation rejected it.

Among the seven keyword positives eligible for the extra source
action, all had at least one target paper in the top 24, and five
had both. The static prototype requirements blocked all five.
One of the seven would satisfy those role requirements on the
true pair, but at least one of its sources was outside the top 24.
The remaining case also lacks top-24 coverage. No correct pair
survived both tests, so wider pairing did not produce more strict
answers at the same per-request forward cap.

These outcomes identify a dependency between source and role
retrieval. A source-only fallback with the existing static role
condition cannot recover the five reachable positive pairs. A
future comparison can use this band as a ranking feature and
retain an explicit final role check under the joint forward budget.
See [aggregate results](results.json).

## Reproduce

Restore the same public inputs and archived cascade reference as in
the [stage audit](../paired-role-stage-audit-v1#reproduce). Use the
same command inside a Slurm GPU allocation, substituting
`--source-postings-reference CASCADE_RUN` for
`--stage-audit-reference CASCADE_RUN`. Repeat with a new output path
and `--reverse`.
