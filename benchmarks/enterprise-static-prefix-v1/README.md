# Reuse a fixed instruction prefix in enterprise qualification parsing

Keep the pinned 80 enterprise development requests, 52,667-source
index and live automatic qualification schedule. Both arms keep
the original BF16 and FP16 clause consensus, independent fallback,
within-request prefix broadcasts, rank guards and evidence
verification. The method changes how request-context key/value
tensors are created.

Before reading any enterprise request, derive the common token
prefix of four distinct artificial inputs. Create separate
resident singleton caches for BF16 and FP16. On a match against
the full freshly rendered request-prefix tokens, fork the
cache metadata and append only the remaining tokens. Appended
keys and values belong to the request. If the fixed tokens do
not match, run the entire prefill. No private request enters
the cross-request cache. Verify the retained prefix length
after every append.

Unchanged token sequences can still produce different scores
under different batching or cache shapes. Measure the parser's
complete categorical labels, mandatory qualification offsets,
candidate order, primary and fallback outputs, and citations
for each request. Rotate route order and report the full warmed
serial pipeline. The paired bootstrap resamples requests.
Startup cache build, model load, queueing and transport are
outside the request timings.

Prefix reuse is an established serving optimization. The
experiment isolates its usefulness for this particular parser
and strong search baseline.
[SGLang, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)

## Results

Slurm job 43381 completed on one RTX 5090 on ws5. It reuses
165 fixed tokens in all 95 prefill calls in the static route,
avoiding 15,675 prefix token computations. The two static
caches use 46.4 MiB of resident tensor storage. Creating
them takes 195.4 ms during startup.

| Route | Parser P50 / P95 | Full search P50 / P95 | Independent parser fallbacks | Issued requests |
| --- | ---: | ---: | ---: | ---: |
| Per-request prefill | 115.5 / 934.9 ms | 175.8 / 991.8 ms | 7 | 37 |
| Fixed prefix reuse | 111.5 / 1,025.1 ms | 171.0 / 1,134.6 ms | 9 | 37 |

Mandatory qualification offsets, retrieved and ranked experts,
and all primary and fallback decisions and citations agree on
80/80 requests. Ten requests have a different categorical
assignment outside the mandatory qualification class. Two also
change independent recheck counts. The full-search P95 point
estimate increases by 142.7 ms, with a descriptive paired
interval of −9.2 to +762.6 ms. The token computation reduction
is accompanied by more conservative parser work. The timing
results give no basis to replace per-request prefill in a
tail-latency-sensitive service. This experiment uses a repeated
development workload and establishes neither bitwise equivalence
nor a population accuracy claim. [Aggregate results](results.json)

## Reproduce

Use the restricted development inputs and pinned models from
[enterprise qualification](../enterprise-shared-qualification-v1).
Run inside Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_static_prefix \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS ROLE_WINDOWS VECTORS.npy \
  VECTOR_METADATA MODELS NEW_AGGREGATE_RESULTS
```
