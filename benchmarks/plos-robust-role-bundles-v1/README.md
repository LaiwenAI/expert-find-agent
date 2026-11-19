# Source-backed recommendations under multiple role interpretations

Keep the publisher snapshot, held phrases and 120-request panel from [the live expert search pilot](../plos-live-expert-bundles-v1/README.md). Use the same frozen Qwen3 embedding and ridge role head. A role is plausible for a clause if its score is within a cutoff of the best score. Estimate cutoffs from the score deficits of the annotated roles on 14 separate calibration phrases. The 80, 90 and 100 labels refer to empirical phrase quantiles. They are not coverage guarantees on composed requests.

A complete interpretation assigns three different roles to the three clauses. Across all admitted interpretations, take the union of required roles. Return an expert only when that expert's own declarations cover the union. The cited bundle is the smallest group of the expert's sources covering it. If the intended roles are among the admitted interpretations, every returned bundle supports them. Multiple interpretations without a common match produce an unresolved result. Compare with the point prediction, the same 80 cutoff with abstention on every ambiguity, and a fixed top-two-per-clause policy.

| Policy | Includes intended role set / 120 | Positive replies | Positive replies with all recommendations supported | Replies with any unsupported recommendation | Unresolved | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Point prediction | 38 | 88 | 35 | 53 | 17 | 8.60 ms |
| Abstain on ambiguity, 80 | 54 | 59 | 35 | 24 | 45 | 8.59 ms |
| Common evidence, 80 | 54 | 75 | 41 | 34 | 29 | 8.60 ms |
| Common evidence, 90 | 54 | 75 | 41 | 34 | 29 | 8.63 ms |
| Common evidence, 100 | 110 | 28 | 28 | 0 | 89 | 8.61 ms |
| Top two, common evidence | 62 | 67 | 37 | 30 | 53 | 8.63 ms |

The widest cutoff produced zero observed unsupported recommendations in 28 positive replies, at the cost of 89 unresolved requests. The 80 cutoff recovered six more fully supported positive replies than abstaining at the same cutoff, alongside ten more replies with unsupported recommendations. Missing the intended interpretation remains the critical error. The 120 composed requests reuse 35 phrases and 16 expert identities. This sample does not establish a population error bound.

The point control reproduces every prior ridge recommendation and exact-response result. Timing includes warm query encoding, hypothesis construction, local declaration reads and bundle serialization. No policy uses a second model call. The result is scoped to structured request lists and publisher-declared roles.

## Reproduce

On a Slurm GPU compute node, use the same publisher, NISO, training and embedding inputs as the linked pilot:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.robust_role_bundles \
  PUBLISHER_FIXTURE NISO_EXAMPLES ROLE_TRAINING EMBEDDING_CHECKPOINT \
  SCRATCH_DIRECTORY NEW_RESULT_JSON
```

[Aggregate results](results.json) include input and script hashes. Per-query responses are excluded from the source release.
