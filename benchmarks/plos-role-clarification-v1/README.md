# One-question relaxation of unresolved expert requests

Replay the 120-request PLOS panel under the widest [common-evidence policy](../plos-robust-role-bundles-v1/README.md). Keep any already resolved response. For an unresolved request, consider a clause only if the admitted assignments disagree on its role. Simulate one free-form reply revealing that clause's intended role, even if the correct role is outside the proposed options. Keep all other clauses unchanged, and rerun the common-evidence rule. The reply is supplied by the benchmark label.

Compare questioning the earliest ambiguous clause, the lowest ridge margin, the largest assignment entropy, and the largest expected number of experts with common evidence. The last policy uses uniformly weighted admitted assignments and resident publisher role unions per expert to score each possible reply. It verifies the actual cited bundle via the shared materialized declarations. Two further policies require a positive reply branch or at least half the modeled branch mass to support a recommendation before asking. These masses are heuristic.

Record questions asked, fully supported positive responses, any unsupported recommendation, correct no-match responses and unresolved cases. A hindsight ceiling tests all eligible clauses with their true labels. Time includes the shared query encode and local verification, the policy decision and the machine's post-answer rerun. Human response time and clarification language are outside the replay.

## Results

| Policy | Questions | Fully supported positive / 120 | Confirmed no match | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| No question | 0 | 28 | 3 | 89 |
| First ambiguous clause | 89 | 59 | 18 | 43 |
| Lowest margin | 89 | 62 | 18 | 40 |
| Highest entropy | 89 | 61 | 18 | 41 |
| Expected expert count | 89 | 62 | 18 | 40 |
| Expected count, any positive branch | 63 | 62 | 17 | 41 |
| Expected count, at least half positive mass | 62 | 62 | 17 | 41 |

On this panel the most selective gate uses 27 fewer questions than asking on every unresolved query, while retaining all 34 additional fully supported positive responses and confirming one fewer no-match case. The hindsight ceiling can recover 34 supported positives and 15 correct no-match responses from the unresolved cohort. Those two ceilings are computed separately. The simpler lowest-margin policy matches the always-ask evidence policy on these counts. No arm produces an observed unsupported recommendation or false no-match in this replay.

The gate run uses an RTX 4090 and has machine-only P95 around 16.1 ms. A first run on RTX 5090 has P95 around 8.8 ms for the ungated policies, which agree on aggregate outcomes. Keep timing comparisons within a run. The 120 composed requests reuse 35 phrases and 16 expert identities. These results measure what an accurate reply could recover. Human answer reliability and service latency need separate evaluation.

## Reproduce

On a Slurm GPU node, use the pinned publisher, NISO and model inputs from the linked benchmark:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.one_role_clarification \
  PUBLISHER_FIXTURE NISO_EXAMPLES ROLE_TRAINING EMBEDDING_CHECKPOINT \
  SCRATCH_DIRECTORY NEW_RESULT_JSON
```

[Aggregate results](results.json) include both runs, checkpoint provenance and input hashes. The first run records its script hash from before the gates were added. Per-request traces stay local.
