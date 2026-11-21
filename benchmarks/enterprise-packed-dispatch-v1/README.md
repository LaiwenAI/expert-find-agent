# Test cross-request verifier packing on original enterprise queries

Transfer the scheduling idea from the
[structured eLife service](../elife-mixed-role-dispatch-v1) to
the [enterprise guarded verifier](../enterprise-packed-role-load-v1).
The frozen 80 development queries enter as original text. The
pipeline interprets qualification spans, encodes the full query,
searches the 52,667-source index for five experts, verifies their
evidence, and runs global role-index recovery on abstention. The
serial reference is the existing guarded implementation.

Both replay paths use the same resumable search stages. Run with
one or up to four already-queued requests on a single GPU, with
no wait to accumulate work. Each request retains its original
query encoding, ranking, verifier minibatches, padding eligibility
rule, and per-batch canonical recheck. Score eligible variable-length
minibatches together across requests with independent attention spans.
Keep padded batches separate. Advance only requests that still need
work and emit each completed response. Search decisions, candidate
and citation order, parsed span boundaries, recovery invocation, and
truncation status are compared byte for byte with independently run
guarded results. Neither sealed requests nor review labels enter the test.

Replay each query once per nominal rate and seed. Pair the exact
offered times and order across capacities, and reverse the execution
order in the second seed. Report observed arrivals, client P95,
completed throughput including drain, actual verifier forwards and
tokens, and changed responses. Report finite runs and timing
conditional on evidence parity. Published results contain aggregate
statistics only.

## Results

Both Slurm tasks in array 46398 finished on RTX 5090 on ws3.
The eight replays completed 640 requests. All 320 paired comparisons
were byte-identical, and each replay issued 37 provisional responses
among 80 requests.

| Arrival seed | Offered requests/s | Client P95 (s), serial → grouped | P95 change | Packed verifier forwards, serial → grouped | Completed requests/s, serial → grouped |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261119 | 2.266 | 4.882 → 5.127 | +5.0% | 111 → 85 | 2.285 → 2.285 |
| 20261119 | 3.297 | 6.079 → 6.660 | +9.6% | 111 → 66 | 2.805 → 2.806 |
| 20261120 | 2.126 | 3.219 → 3.591 | +11.5% | 111 → 76 | 2.025 → 2.024 |
| 20261120 | 3.093 | 4.623 → 4.709 | +1.9% | 111 → 65 | 2.631 → 2.629 |

The grouped worker reduces physical packed forwards by 23%–41%.
Total verifier tokens are unchanged in three comparisons and 1.0%
lower in the fourth because one padded recheck is skipped. Completed
throughput changes by at most 0.002 requests/s. Client P95 rises by
1.9%–11.5% in all matched comparisons. Grouped execution advances
earlier requests only after the group reaches the next verification
wave. Lower dispatch count alone is insufficient to improve
end-to-end service latency here.

Keep serial guarded execution as the enterprise comparison baseline.
This is a finite replay of previously analyzed development traffic
on one GPU and tests output preservation, not expert suitability.
It establishes a transfer limit for the tested scheduling policy.
See [aggregate results](results.json).

## Reproduce

Restore the [guarded baseline inputs](../enterprise-packed-role-load-v1#reproduce).
Use one Slurm CUDA allocation per seed and a fresh output. Repeat
with the other seed and `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_packed_dispatch \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS PRIVATE_ROLE_WINDOWS PRIVATE_VECTORS.npy \
  PRIVATE_VECTOR_META MODELS benchmarks/enterprise-packed-role-search-v1/results.json \
  NEW_RESULTS.json --seed 20261119
```
