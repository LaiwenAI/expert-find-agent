# Test packed role recovery in a queued enterprise search service

Reuse the exact development requests, source vectors, role windows,
model weights, packed kernel and guard from the
[serial packed experiment](../enterprise-packed-role-search-v1).
The same reader is used for base verification and on-abstention
role recovery. Run three arms with one FIFO GPU worker behind
a loopback HTTP service.

| Arm | First-pass and rescue execution |
| --- | --- |
| Padded whole | Canonical padded verifier and no rescue |
| Padded rescue | Canonical padded verifier with global role-index rescue on abstention |
| Guarded rescue | Same recovery policy with selective packed attention and canonical rechecks near acceptance |

Replay two independently seeded open-loop Poisson arrival sequences
at a nominal 3 requests per second. Each trace visits all 80
development requests once. Pair arrivals and request order across
arms, reverse arm order for the second seed, and exclude exactly
one warm-up request per arm. Client latency is measured from
scheduled arrival to decoded response. Also report observed rate,
queue and service quantiles, and final drain. This is a single-host
queue test with overlapping arrivals and one serial GPU worker.
It does not emulate a distributed or multi-worker production system.

Compare exact response bytes, which include qualifications, ordered
initial retrieval, each primary decision and citation, all fallback
citations, and fallback invocation. Count executed tokens including
temporary padding or duplicate guard rechecks. Provisional outputs
are from the frozen verifier. No sealed query or independent human
judgment is used.

## Results

Slurm job 43156 on one RTX 5090 completed all 480 attempted
responses. Both traces produce byte-identical responses between
guarded and padded recovery, including all three extra provisional
recommendations per trace.

| Offered requests/s | Arm | Issued / 80 | Client P95 (s) | Queue P95 (s) | Service P95 (s) | Completed/s including drain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3.091 | Padded whole | 34 | 6.410 | 6.010 | 1.130 | 2.633 |
| | Padded rescue | 37 | 6.864 | 6.489 | 1.158 | 2.577 |
| | Guarded rescue | 37 | 5.847 | 5.469 | 1.084 | 2.718 |
| 2.900 | Padded whole | 34 | 4.919 | 4.019 | 1.128 | 2.453 |
| | Padded rescue | 37 | 4.991 | 4.137 | 1.156 | 2.436 |
| | Guarded rescue | 37 | 4.562 | 3.653 | 1.079 | 2.512 |

Guarded recovery reduces observed client P95 relative to padded
recovery by 14.8% and 8.6%. Against padded whole, it reduces
observed P95 by 8.8% and 7.3% while retaining the three extra
provisional results. Each guarded trace executes 421,158 verifier
tokens including the canonical rechecks. Padded recovery executes
585,690 and padded whole 544,236.

The traces finish with a nonzero queue drain. Completion rates here
are finite-trace rates, not estimates of stable capacity at these
arrival levels. The two seeds replay the same development set and
do not supply independent evidence of expert suitability or
generalization to new requests. The additional recommendations
still require human adjudication.

[Aggregate results](results.json)

## Reproduce

Use the pinned private projections, GPU model revisions and dense cache
from the serial benchmark. Run in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_packed_role_load \
    DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS PRIVATE_ROLE_WINDOWS PRIVATE_VECTORS.npy \
    PRIVATE_VECTOR_META MODELS benchmarks/enterprise-packed-role-search-v1/results.json NEW_RESULTS.json
```
