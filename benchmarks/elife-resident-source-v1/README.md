# Confirm source identity from resident retrieval features

Keep the dense same-person source proposal and use the remaining
forward for the frozen 4B role reader, which separated the two held
fallback outcomes in the [two-forward diagnostic](../elife-mixed-role-verifier-v1).
Evaluate whether an inexpensive source gate can replace the source
forward under this budget.

Fit two ridge models to identical candidate rows from the frozen PLOS
projection. Use the five existing source features from
`engineering_joint_calibration.features` on the top 24 candidates with
the target present or absent. They derive from resident lexical and
dense ranking. The gold model uses the exact document label. The
distilled model averages this label with a temperature-3 sigmoid of
the frozen 0.6B source verifier margin relative to its cutoff.
Group calibration by clue article into four hash folds. Freeze the
highest out-of-fold missing-source score as the cutoff, then refit
coefficients on all calibration rows. Out-of-fold risk and training
coefficients remain distinct.

On each fallback, both proposed sources must pass the source gate.
The sole model forward then resolves both role clauses in a 4B batch.
The proposed author must still have both corresponding declarations.
No extra source forward is used. Compare with archival service proofs
on the 27-pair development and 12-pair held eLife panels. For the
missing-source comparison, remove each target in turn and count only
new fallback outputs, keeping the original source selection and the
shared forward cap. Compare two request orders. These are component
results on previously examined panels, without live-service timing.
For diagnosis, pool the candidate-pair margins from the two panels.
Count intended pairs that score strictly above every observed wrong
pair in the same query view. This is a hindsight bound on this
candidate set, with no evaluation-time threshold update.

## Results

Both request orders per panel complete on ws3 RTX 5090 in Slurm
array 46316. All output proof signatures reproduce, including the
archived service controls and the initial runs in array 46310.

| Source head | PLOS keywords passed | PLOS excerpts passed | PLOS missing-source rows passed per view | eLife available fallbacks admitted |
| --- | ---: | ---: | ---: | ---: |
| Gold features | 3/19 | 8/24 | 0/576 | 0/6 |
| Verifier-distilled features | 2/19 | 7/24 | 0/576 | 0/6 |

Five of the six available eLife fallbacks have the intended source
pair. Neither source gate admits a pair, so there are no extra role
calls or emitted proofs. The development and held keyword counts
stay at 12/27 and 10/12. With target documents removed, the
development replay proposes eight fallback pairs and rejects all
eight. The held replay presents no fallback proposal under removal,
so its abstentions provide no classifier test for missing sources.

Across both panels, all five correct keyword proposals score higher
than the three wrong keyword proposals for each head. The calibrated
generic cutoff sits above all five correct pairs. This hindsight
comparison indicates a calibration-transfer bottleneck on the
observed candidate set and supplies no new operating threshold.
Estimate risk on a larger set of same-author near-neighbor proposals
before changing the rule. See [aggregate results](results.json).

## Reproduce

Restore both eLife projections, PLOS calibration, and corresponding
single-run references from the
[mixed verifier experiment](../elife-mixed-role-verifier-v1#reproduce).
Replace its diagnostic flag with
`--resident-source-reference MATCHED_SERVICE_RUN`.
Use both request orders in fresh Slurm GPU allocations.
