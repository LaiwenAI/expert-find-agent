# Share prefix storage across first-query qualification checks

The existing [cross-precision first-query pipeline](../enterprise-cross-precision-search-v1)
reuses request tokens but copies the BF16 or FP16 prefix cache and repeats it
across the suffix batch. This comparison replaces the copy and repeat with
a read-only batch broadcast of each prefix tensor. Each decoder call receives
new cache metadata and a separate dynamic layer wrapper. Appending the suffix
still uses the installed Transformers concatenation and attention code. The
implementation shares neither appended suffixes nor data across requests.

The two arms share the resident model weights, the original numerical
recheck policy, and the same exact index and evidence verifier. Both evaluate
all 80 previously used development requests against 52,667 owned sources.
Alternating order, compare all clause labels, qualification offsets, retrieved
expert order, final ordered recommendations, witness offsets, unresolved
states, and the number of independent fallback blocks. No sealed requests
enter the experiment.

The timing window runs from clause splitting through the final ranking.
GPU allocator peak is measured relative to memory already allocated before
each call. Index construction, model loading, transport and scheduling queues
are excluded. Count the logical bytes of per-batch prefix replicas the
broadcast avoids. The final key/value concatenation still produces a dense
attention input; the byte count does not measure GPU bandwidth or eliminate
this later materialization. Prefix sharing has established implementations at
the attention layer. [DeFT, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a6df53f082619d02b9fad64a022e5de3-Abstract-Conference.html)

## Results

Slurm job 42927 ran both arms on a single RTX 5090. No request changes any
clause label, qualification span, candidate order, final evidence decision,
citation offset, or independent fallback count. Both arms reproduce the
published development reference totals: 1,280 criterion–source checks, 70
provisionally supported experts and seven independent parser fallbacks.

| Route | Parser median / P95 | Complete search median / P95 | Extra live allocator peak P95 |
| --- | ---: | ---: | ---: |
| Copy and repeat | 121.73 / 997.57 ms | 194.46 / 1,181.04 ms | 2.383 GiB |
| Broadcast shared storage | 119.42 / 953.45 ms | 191.98 / 1,137.28 ms | 2.383 GiB |

Full first-query P95 is 43.76 ms lower (3.7%). The descriptive paired
request-bootstrap interval for broadcast minus reference is −45.22 to
−6.82 ms. The 243 branches avoid materializing about 264 GiB of logical
prefix replicas summed over the run. Peak allocated memory is effectively
unchanged. Suffix concatenation and full-block fallback still materialize
large working sets. The result supports an incremental under-the-hood
speedup. Repeated development queries and model-output equivalence cannot
establish human matching quality or a serving-load gain.

A separate BF16 cache probe on the
[72 public synthetic controls](../enterprise-heading-roles-v1/controls.json)
compares original cached and broadcast execution with identical tokens per
block. All 96 blocks have bit-identical logits (Slurm job 42937). [Public probe results](public-cache-check.json)

[Aggregate development results](results.json)

## Reproduce

Use the source projection and checkpoints from the
[first-query comparison](../enterprise-cross-precision-search-v1). Run in a
Slurm CUDA allocation with Transformers 5.23.0.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_prefix_fork \
  DEVELOPMENT_QUERIES DEVELOPMENT_SOURCES CHECKPOINT_DIRECTORY NEW_RESULTS.json
```

For the public cache probe:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.verify_enterprise_prefix_fork \
  QWEN_4B_CHECKPOINT benchmarks/enterprise-heading-roles-v1/controls.json NEW_PROBE.json
```
