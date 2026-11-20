# Resolve an ambiguous INT8 title boundary with local FP32 scoring

The [bounded INT8 guard](../real-title-guard-v1) recovered all FP32
responses on SciRepEval. Its full-index fallbacks raised P99.
Keep the interval scan and the double-precision recheck unchanged.
That scan retains every candidate that can cross the FP32 cutoff
under the original numeric assumptions. Change only how the remaining
ambiguous rows are evaluated.

Compare two alternatives to a full FP32 fallback. Gather the exact
rows into one compact matrix, or run the FP32 matvec on every
aligned 32-row block touched by the band. The latter preserves the
source order and contiguous input layout within each block. Use only
scores of rows in the original band and stable source-ID tie-breaking.
This experiment tests whether the actual FP32 scoring implementation
returns identical values on the frozen host when the matrix shape
changes. Numerical equivalence of arbitrary BLAS builds is outside
the claim.

For every held-out query, compare the bit patterns of all in-band
scores under both local methods against full FP32. Count differences
on the subset of bands that trigger the fallback separately. Then run
both local alternatives, the full-fallback guard and the FP32 baseline
through the same warm title-to-local-citation path. Measure cached and
online query modes, three interleaved repetitions, identical output
bytes and P95/P99. Loading original vectors from a warm memory map
still requires a full backing file. The replay excludes semantic
verification, service queues, and cold storage behavior.

## Results

Jobs `43869` and `43871` match all non-timing results. On ws5, NumPy 2.5.5 reports scipy-openblas 0.3.34.237.2.
Across 91 unique query vectors, gathering the band changes 75 of
1,323 FP32 score bit patterns, including four of 98 in the
ambiguous bands. Aligned 32-row computation changes none. All
paths return the same ordered citation bytes as the full FP32
reference for all 273 paired runs in both modes. Each guarded
path uses its recheck in 12 instances per mode.

| Warm time, job 43869 | FP32 | Full fallback | Gathered FP32 | Aligned 32-row FP32 |
| --- | ---: | ---: | ---: | ---: |
| Cached P95 | 6.20 ms | 1.63 ms | 1.68 ms | 1.43 ms |
| Cached P99 | 6.41 ms | 6.99 ms | 1.88 ms | 1.66 ms |
| Online P95 | 14.67 ms | 11.51 ms | 10.57 ms | 10.02 ms |
| Online P99 | 15.24 ms | 15.83 ms | 10.91 ms | 10.61 ms |

Contiguous scoring removes the observed penalty of the full
fallback while preserving bitwise evidence for the reference arithmetic
on this host. The gathered alternative preserves responses here
despite score differences, so exact output alone would have hidden
its arithmetic incompatibility. The contiguous path remains an
experimental optimization tied to a validated backend and shape.
The original full fallback stays available as the conservative
reference. Cold-file, cross-platform and loaded service timing are
out of scope.

[Aggregate replay](results.json)

## Reproduce

Use the [pinned SciRepEval archive and checkpoint](../real-title-runtime-v1)
with Slurm CUDA and AVX-512 VNNI on the original host. Use a fresh library
path and output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.real_title_block_recheck QWEN_EMBEDDING \
  NEW_TEMP_LIBRARY NEW_RESULTS benchmarks/real-title-guard-v1/results.json
```
