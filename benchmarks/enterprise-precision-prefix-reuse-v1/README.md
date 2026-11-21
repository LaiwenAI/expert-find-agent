# Reuse request-prefix computation in qualification rechecks

The [selective precision reader](../enterprise-cross-precision-v1)
runs a cached BF16 pass. Blocks near the qualification boundary
trigger a second precision. That path independently runs an FP16
forward over the full request prefix, then over the suffix batch.
The [broadcast-prefix optimization](../enterprise-prefix-fork-v1)
already avoids copying repeated cache rows within each precision.

Keep the released model, original clause prompts, token order,
block geometry and independent BF16 fallback fixed. Test whether
the FP16 suffix can read a converted copy of the existing BF16
keys and values. Conversion preserves the source tensor, sequence
positions and metadata. The result still starts from activations
computed in BF16. It can change the qualification decision of
the FP16 check.

| Policy | Secondary prefix computation |
| --- | --- |
| Native | Independently compute the FP16 prefix when the existing gate requests a second opinion |
| Cast | Convert the existing BF16 cache to FP16 and keep the original secondary margin rule |
| Guarded cast | Accept a converted-cache check only if all qualification decisions agree with BF16 and all converted-cache absolute margins exceed 1.5. Otherwise run the native FP16 check and reuse that native prefix for remaining blocks |

The margin of 1.5 is fixed for this study, and agreement between
the converted and BF16 branches does not imply independence.
The guarded policy can miss a native disagreement. Record the
frequency of that event in the stress test along with changes to
whole-request qualification offsets in the natural replay.

Measure warm parser time on the pinned 80 development requests
and 72 bilingual heading controls. Rotate method order and
reverse request traversal on the second run. Paired intervals
resample the observed requests, and timings exclude retrieval,
verification, model loading, and serving overhead. The enterprise
comparison is with a model reference. The 72 authored public
target labels are assessed separately.

Also create 24 public long-context cases by adding unrelated
archival context as in the [layout stress test](../enterprise-recheck-layouts-v1).
Force the target block through both secondary-cache variants.
Compute whether each alternative would trigger the whole-block
BF16 recheck, and compare qualification marks and the anchored
target label. This probes numeric behavior outside the natural
trigger distribution.

## Results

Both RTX 4090 tasks in Slurm array 47461 on ws6 reproduce every
clause decision under reversed execution. The qualification changes
below are differences from the established native-precision parser.

| Development route | Native FP16 prefix forwards | FP16 suffix forwards (cast + native) | Whole BF16 rechecks | Changed qualification sets / 80 | Warm parser P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 14 | 24 | 9 | 0 | 1,285–1,318 |
| Cast | 0 | 24 | 7 | 2 | 1,211–1,241 |
| Guarded cast | 11 | 35 | 9 | 0 | 1,330–1,361 |

Direct conversion skips two native rechecks, each on a request whose
qualification set changes. The guarded route preserves all four-way
clause labels on these 80 requests. It saves three native prefix
forwards while adding eleven FP16 suffix forwards. The P95 point
estimate increases by 43–45 ms, with descriptive paired intervals
of −76 to +63 ms and −105 to +63 ms. The direct route's P95
intervals also cross zero. This replay gives no latency advantage
for a route that keeps the observed qualification decisions.

All routes classify the same 61 of 72 original public targets
correctly. On the 24 forced long-context blocks, direct conversion
triggers one additional recheck, which changes one target from
wrong to correct. Guarded conversion falls back to native there
and matches its 21 correct targets. The forced panel happens to
contain no native whole-block fallbacks. It cannot estimate the
guard's ability to preserve native rechecks on new requests.
Retain the native secondary prefix in the reference parser.

[Aggregate runs](results.json).

## Reproduce

Use an allocated Slurm CUDA device, the checkpoints and pinned
development queries from the linked benchmarks, and new output
paths. Repeat with `--reverse`.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_precision_prefix_reuse \
  DEVELOPMENT_QUERIES benchmarks/enterprise-heading-roles-v1/controls.json \
  QWEN3_4B_CHECKPOINT NEW_AGGREGATE.json
```
