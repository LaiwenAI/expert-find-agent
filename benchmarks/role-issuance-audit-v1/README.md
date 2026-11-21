# Audit which stage produces adverse proof outputs

Reuse the frozen [eLife prompt holdout](../elife-prompt-holdout-v1)
and the models used by the [HTTP replay](../elife-role-prompt-service-v1).
Instrument the result before issuing it. Record whether it was
supported by the initial role bands or created by the single-clause
recheck. With the labeled cases, check source identity first,
then whether both intended roles are in the final proof, and
then whether extra contributors were issued. For role errors,
identify whether the reviewed clause, the untouched clause, or
both lacked the intended role. Classify outputs only after search.
The role and source labels are unavailable to the decision path.

Evaluate three fixed guards on the name-only path. The first
requires both selected sources to lead their own fused rankings.
The second requires the whole union of ridge and prototype
role sets on any untouched clause, instead of just the ridge
roles. The selected revised clause retains the independently
checked single role. The third combines both guards. All guards
filter proposed results after the original retrieval and recheck,
including proposals made without a reader call. No thresholds
are fitted to this holdout. The extra reader call count cannot
improve from filtering at this stage.

Score excerpt and keyword clues with both targets visible, and
hide each target in turn as before. These experiments diagnose the
same holdout used for the accuracy and HTTP replay comparisons.

## Results

With both target sources available, the original outcomes reproduce
the prior holdout and HTTP results. The two clue types are separate
views of the same 13 paper pairs.

| Clue | Policy | Proofs / 66 | Adverse / issued | Error stage |
| --- | --- | ---: | ---: | --- |
| Excerpt | Static prototype | 35 | 4/40 | Initial role × 4 |
| Excerpt | Full definitions | 41 | 6/47 | Reviewed role × 6 |
| Excerpt | Role names | 42 | 6/48 | Reviewed role × 6 |
| Excerpt | Any of the three added guards | 42 | 6/48 | Reviewed role × 6 |
| Keywords | Static prototype | 26 | 1/27 | Initial role × 1 |
| Keywords | Full definitions | 32 | 1/33 | Reviewed role × 1 |
| Keywords | Role names | 33 | 1/34 | Reviewed role × 1 |
| Keywords | Any of the three added guards | 33 | 1/34 | Reviewed role × 1 |

For both reader prompts, all seven adverse results involve the one
reinterpreted role, with the designated paper pair and the intended
role retained on the untouched clause. None is attributed to source
substitution, a misread untouched clause, or a contributor error with
both intended roles present. All routes have zero substitutions in
the 264 missing-source views per clue type. The excerpt test includes
one false issuance on a swap per route.

Tighter checks on the source pair and untouched clause leave the
measured outcomes unchanged. On these cases, the independent
prototype band accepts the same incorrect revised role. Development
should test a stronger contrast between confusable roles on the
selected clause and measure strict proofs and risk on new wording
and sources. Aggregate counts alone do not identify whether the
two prompts fail on all the same variants.

Both tasks of Slurm array 45976 completed on ws3 RTX 5090. All
non-timing fields match with reversed requests. Timing factors in
local retrieval and the measured reader call. It excludes transport
and unrestricted query parsing. These repeats audit the failure
stage rather than provide new independent labels.
[Reproducible outcome counts](results.json)

## Reproduce

Restore the eLife projection from its upstream holdout manifest.
Use the pinned model and band artifacts. Run inside Slurm CUDA,
then repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.role_issuance_audit \
  ELIFE_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```
