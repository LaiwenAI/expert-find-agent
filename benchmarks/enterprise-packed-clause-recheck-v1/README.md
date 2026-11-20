# Pack all clauses in triggered enterprise parser blocks

The [selective clause comparison](../enterprise-clause-recheck-v1)
reduces fallback work but changes mandatory qualification
decisions. Here each triggered block still recomputes every
clause against the complete request. Compare the original
padded independent batch, a packed block, and a guarded
packed block.

Reuse the [isolated packing](../enterprise-isolated-parsing-v1)
implementation with the same BF16 weights and token IDs.
The packed forward represents the request once. Its attention
mask gives each clause suffix access to that common request
and its own preceding tokens. Position IDs match each
independent sequence. Passing all suffixes through packing can
still change scores because tensor shapes and numeric kernels
differ.

The guarded route executes the original independent batch if
the packed qualification margin is at most 2.0 anywhere in
a triggered block, or qualification membership disagrees with
either cached precision. It counts both forwards when the guard
fires. The three arms keep the same preceding cross-precision
trigger and the automatic downstream evidence schedule.

Run the pinned 80 private development searches and 72 public
synthetic parser controls. Compare every clause, qualification
offset, issued expert and source span. Measure the full warm
pipeline on a single GPU, with rotated route order and paired
request bootstrap. The fallback slot count measures input
shape and does not track actual FLOPs or memory traffic.
Serving queues and independent suitability judgments are
outside the replay.

## Results

Slurm job 43394 completed on one RTX 5090 on ws5. Each
arm reaches seven triggered blocks with 56 clause rows.

| Route | Fallback token slots | Canonical blocks after packing | Parser P95 | Full search P95 | Requests with changed qualifications |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole block | 74,680 | — | 939.3 ms | 996.5 ms | 0 |
| Packed block | 12,018 | 0 | 636.4 ms | 712.7 ms | 2 |
| Guarded packed block | 86,698 | 7 | 946.4 ms | 1,005.9 ms | 0 |

The packed route reduces the input slot count by 83.9% and
the full P95 point estimate by 283.8 ms. Its descriptive
paired difference interval is −823.2 to +1.3 ms. It changes
qualification offsets and primary verification details on
two requests. All three routes issue the same experts and
citations on the same 37 requests. The public controls retain
the same 60/72 correct anchored targets, but one control has
a different qualification elsewhere under unguarded packing.
The guard preserves all observed labels by recomputing all
seven original blocks, so it spends more input slots than the
control. Keep the original whole-block fallback as the
output-preserving baseline. The experiment identifies a
substantial fast-path opportunity and a guard that fails
to exploit it under the observed risk conditions.
[Aggregate results](results.json)

## Reproduce

Use the authorized development data, checksums, and model
setup from the [selective clause comparison](../enterprise-clause-recheck-v1).
Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_packed_clause_recheck \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS ROLE_WINDOWS VECTORS.npy \
  VECTOR_METADATA MODELS benchmarks/enterprise-heading-roles-v1/controls.json NEW_RESULTS
```
