# Replay mandatory-criterion screening under queued enterprise load

Compare the guarded fixed-order schedule and the automatic
objective-first schedule from the [serial enterprise study](../enterprise-shared-qualification-v1).
The latter stages requests only when the original verification
would have at least 32 pairs and more than one criterion.
Reuse the same pinned 80 development requests, 52,667-source
index, parser, embeddings, role window fallback and models.

Replay two Poisson arrival traces at each nominal rate, 1.8 and
2.7 requests per second. The lower rate uses one randomized
pass over the 80 requests per seed. The higher rate uses two
separately shuffled passes. Both arms receive the same requests
at the same offsets per trace. Counterbalance arm order. A
single FIFO worker serves loopback HTTP. Each arm has one
excluded warm-up and reruns parse, retrieval and verification
for every timed request. Each service instance drains before
the next arm starts.

Measure arrival-to-response, queue and service latency, and
completion rate including the drain. Check early and late
halves to inspect queue growth. Report each trace separately
because nearby requests share queue state. The planned rates
are nominal. Poisson realizations give different observed
offered rates. The finite traces provide workload evidence
at those rates. Stable production capacity requires a longer
stationary service study with varied requests and load.

Compare the ordered issued experts and citations, fallback
results, retrieval order, parsed criteria, and rescue
invocation for every pair of responses. Compare per-criterion
details through in-memory digests, since early elimination can
leave some rejected candidates' criteria unresolved. Publish
only aggregate observations. The replay uses the development
cohort and evaluates execution, with expert suitability
pending independent judgment.

## Results

Slurm job 43297 completed 960 responses, 480 paired requests
across the four traces, on one RTX 5090 on ws5. The issued
expert order, source spans, fallback outputs and other
compared response fields match in all 480 pairs. Twelve
pairs differ in partial criterion details for rejected
candidates. Issued coverage is 37 per 80 requests in both arms.

| Observed offered/s | Requests per arm | Client P95, guarded → automatic (s) | Queue P95 (s) | Service P95 (s) | P95 improvement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.740 | 80 | 3.204 → 3.128 | 2.877 → 2.801 | 1.040 → 0.969 | 2.4% |
| 1.629 | 80 | 4.013 → 3.789 | 3.241 → 3.016 | 1.036 → 0.966 | 5.6% |
| 2.891 | 160 | 8.095 → 7.758 | 7.676 → 7.319 | 1.075 → 1.001 | 4.2% |
| 2.574 | 160 | 2.583 → 2.521 | 2.255 → 2.189 | 1.075 → 1.000 | 2.4% |

The automatic route executes 380,257 verifier tokens per 80
requests, versus 421,158 for guarded batching, a 9.7% reduction.
Service P95 falls by 6.8% to 7.0% across these traces. Client
P95 falls by 2.4% to 5.6%. Completed throughput including drain
changes by at most 0.01 requests/s.

At 2.891 offered requests/s, queue P95 rises between the first
and second halves for both arms. At 2.574 offered requests/s,
it falls for both. The timings support a consistent observed
latency improvement across the paired traces. Workload order,
realized arrival rate and final drain still matter. The study
uses repeated development queries in one loopback service,
and it leaves stable capacity and independent relevance open.

[Trace aggregates](results.json)

## Reproduce

Use the private serial experiment inputs and result provenance
with the existing variable-length reader on Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_shared_qualification_load \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS ROLE_WINDOWS VECTORS.npy VECTOR_META \
  MODELS SERIAL_RESULTS NEW_AGGREGATE_RESULTS
```
