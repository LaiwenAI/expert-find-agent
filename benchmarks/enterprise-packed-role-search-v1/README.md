# Evaluate packed verification in live enterprise first search

Reuse the pinned 80 development queries, 52,667 sources, exact dense
cache and 5,734 role windows from the
[live role-index experiment](../enterprise-role-first-query-v1).
Each timed request starts with the fast shared-prefix qualification
parser and performs fresh query encoding, dense retrieval, primary
verification and result construction. At most two role windows are
retrieved and verified when primary verification yields no result.
This remains a warm serial experiment.

| Arm | Primary verification and fallback |
| --- | --- |
| Padded whole | Canonical left-padded reader, without role fallback |
| Padded rescue | Same reader with global role-index fallback on abstention |
| Packed rescue | Same fallback, shared-weights packed causal attention for multi-row batches |
| Guarded rescue | Same fallback, pack when the unpadded token count is below 80% of padded shape and repeat the original padded batch if any packed margin lies within 0.125 of zero |

The input prompt, tokenization, truncation and answer labels are
unchanged. Reuse the independent-sequence CUDA implementation from
the [CoF comparison](../cof-ragged-frontier-v1), with shared read-only
parameters and a separate attention configuration. The guard
addresses scores near acceptance. It is not a proof of exact
citation selection because competing positive scores can move without
crossing zero. Record observed changes to all candidate decisions,
ordered initial citations and exact rescue outputs. Queueing, cold
start and independent expert relevance judgments are outside scope.

Report actual model work across both phases. Logical tokens count
each prompt position once. Executed tokens count padded shapes,
packed real tokens, and any duplicated work on guard rechecks.
The experiment rotates arm order per query. P95 intervals resample
paired requests.

## Results

Slurm job 43148 ran all 80 requests on one RTX 5090 on ws5.
The padded recovery arm reproduces the earlier recommendations
and exact citations. All four arms share identical initial retrieval
and qualification spans.

| Arm | Requests with provisional output | Executed verifier tokens | P95 for primary verifier (ms) | Live P95 (ms) | Requests with changed initial evidence or recommendations vs padded rescue |
| --- | ---: | ---: | ---: | ---: | ---: |
| Padded whole | 34 | 544,236 | 179.47 | 1126.37 | 0 |
| Padded rescue | 37 | 585,690 | 179.25 | 1155.29 | 0 |
| Packed rescue | 38 | 331,645 | 96.62 | 1071.86 | 5 |
| Guarded rescue | 37 | 421,158 | 115.00 | 1076.46 | 0 |

The guard triggers 16 padded rechecks. Guarded and padded rescue
make the same 1,464 logical criterion comparisons and reproduce every
primary decision, recommendation, citation and fallback output.
Guarded execution saves 28.1% of executed verifier tokens, including
the rechecks. Its paired live P95 difference versus padded rescue
has a descriptive 95% interval of −248.86 to −0.95 ms.
Guarded rescue has a point P95 below the no-fallback baseline, but
the paired interval against that baseline is −187.19 to +19.93 ms.
The interval is calculated from the same serialized request timings.

Unguarded packing changes five initial recommendation or citation
sets, including one request that newly receives an initial output
instead of invoking the fallback. Acceptance and citation equality
for the guarded policy is observed on this development cohort.
This experiment does not independently validate either the base
results or the three additional model-supported recommendations,
or estimate effects under concurrent service load.

[Aggregate results](results.json)

## Reproduce

Reuse the private inputs and checkpoints from the linked live
benchmark. Run on Slurm CUDA with the variable-length attention
backend compatible with the pinned reader.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_packed_role_search \
    DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS PRIVATE_ROLE_WINDOWS PRIVATE_VECTORS.npy \
    PRIVATE_VECTOR_META MODELS NEW_RESULTS.json PRIVATE_TRACE
```
