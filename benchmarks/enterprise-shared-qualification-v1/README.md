# Reuse exact verifier inputs and prune failed prerequisite checks

Test two changes to the existing guarded packed verification
stack on the pinned 80 private development requests. Use the
same clause parser, query encoder, 52,667-source dense index,
top-five experts, two sources per expert and abstention-triggered
role-index fallback. The benchmark emits aggregate results only.

| Arm | Verification schedule |
| --- | --- |
| Guarded | Original fixed-order batching with guarded packing |
| Shared | Score each identical criterion/excerpt text pair once within the request |
| Qualification first | Share exact pairs and test the mandatory search objective before screening further qualifications |
| Automatic | Use objective-first scheduling only if the request has at least 32 primary pairs and a further qualification |

When two people have an identical model input, reuse the
model score while retaining each original owner, source ID,
and span. Model computation is shared, and each citation still uses
its original corpus association for person and source. When neither
eligible source passes the mandatory objective, that candidate
cannot satisfy the conjunction. Other criteria need no additional
model comparisons for that candidate. Keep their skipped positions
unresolved. The fallback still receives the original ranked experts.

Pre-encode each unique prompt with the original context cap.
Return to the original guarded schedule if any input truncates,
a computed margin is within 0.125 of zero, or two different
positive citations compete within a margin of 0.125. This
preserves the usual checks on clipped requests and addresses
numerical changes caused by different batch shapes. The margin
rule is a heuristic. Measure equality directly for all observed
outputs. Count primary and fallback executed tokens, including
rescore work. Order of input arms rotates across requests.

Compare issued experts and citations, rescue outputs, rank,
qualifications and whether the rescue was invoked. Compare
all per-criterion decisions separately because objective-first
execution can suppress partial evidence for rejected candidates.
Timing includes live parse, embedding, retrieval, primary and
rescue verification and aggregation. It excludes initialization,
queueing, and blind relevance judgments.

## Results

Slurm job 43289 ran all 80 requests on one RTX 5090 on ws5.
All four schedules issued provisional recommendations on the
same 37 requests. They reproduced issued expert order and
citations, fallback outputs and invocation, retrieved experts,
and parsed qualifications for every request.

| Arm | Staged requests | Canonical rechecks | Executed verifier tokens | Primary verifier P95, ms | Live P95, ms | Changed partial decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Guarded | 0 | 0 | 421,158 | 114.59 | 1067.86 | 0 |
| Shared | 0 | 1 | 420,462 | 117.61 | 1073.79 | 0 |
| Objective first | 25 | 6 | 359,892 | 125.97 | 996.76 | 3 |
| Automatic | 7 | 2 | 380,257 | 110.24 | 996.40 | 2 |

Exact sharing finds 14 duplicate inputs among 1,280 logical
primary rows. Its executed-token savings are below 0.2% after
one canonical recheck. Objective-first schedules reduce executed
tokens by 14.5% and 9.7% in the always-eligible and automatic
arms, respectively. Only the automatic arm also lowers the
point estimate of the primary-verifier P95. Its complete
P95 falls by about 71 ms, with a paired 95% interval of −121
to +58 ms. The experiment establishes a reduction in model
work on this workload. Tail latency remains inconclusive.

The automatic path changes partial criterion decisions on
two rejected requests. Issued recommendations and citations
are identical. The changed partial states matter if a client
shows all supporting evidence for rejected candidates.
Independent quality judgments and serving under load remain
outside this comparison.

[Aggregate results](results.json)

## Reproduce

Use the restricted inputs from [enterprise role first query](../enterprise-role-first-query-v1)
and the pinned packed-reader implementation inside Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_shared_qualification \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS ROLE_WINDOWS VECTORS.npy \
  VECTOR_METADATA MODELS NEW_AGGREGATE_RESULTS
```

Keep all requests, source files, role windows and vectors private.
