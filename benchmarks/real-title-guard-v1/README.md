# Preserve FP32 citations with a bounded recheck of INT8 title scores

Use the unchanged [73,257-source SciRepEval replay](../real-title-runtime-v1).
The original INT8 path altered two ordered responses. This experiment
keeps a compact resident scan index and maps original vectors from a
local FP32 backing file for selective full-precision scoring.

For source vector x and query vector q, let xhat and qhat be their
scaled INT8 reconstructions. Bound the absolute score difference by
the sum of the source residual norm times the query norm and the
reconstructed source norm times the query residual norm. Store one
upward-rounded residual norm per source. Use a global upper bound
for reconstructed source norms. Add conservative allowances for the
FP32 dot product and double-precision rounding. Each fast VNNI score
then gives a lower and upper bound on the bounded FP32 reference.

Take the second-largest lower bound as the threshold. Any row with
an upper bound below it cannot be in the FP32 ordered top two. Read
only the resulting band of rows from the FP32 memory map and score
them in double precision. When the best, second-best and third-best
bands overlap under the reference rounding bound, or the top two
cannot be strictly ordered, rescore with the original full FP32 path.
Respect the reference source-ID tie rule on that path. Synthetic
checks cover negative scores, exact duplicate vectors and different
vector widths.

Compare bounded FP32, raw INT8 and guarded INT8 in cached and online
query modes. Use the same SQLite citations and three interleaved
repetitions per request. Require byte equality for every guarded
response with the corresponding FP32 result. Validate raw INT8
differences against the previous replay. Report the width of the
band and the count of full fallbacks.

The figures are warm local latencies, including citation reads and
serialization. Report the compact payload and the full FP32 backing
file separately. Operating-system page cache residency and cold
storage effects are outside the measurement. No semantic verification,
role screening, network or request queues are added.

## Results

Across jobs `43859` and `43862`, all 273 guarded responses in each query mode are
byte-identical to FP32. Raw INT8 matches 267 of 273, reproducing
the prior unguarded result. Median and P95 recheck widths are
12 and 36 rows, with a maximum of 48. There are 12 full FP32
fallbacks per mode. The compact VNNI scan payload is 75,894,252
bytes. The separate FP32 backing holds 300,060,672 vector bytes.

| Warm latency on job 43862 | Bounded FP32 | Raw INT8 | Guarded INT8 |
| --- | ---: | ---: | ---: |
| Cached P95 | 6.18 ms | 1.70 ms | 1.77 ms |
| Cached P99 | 6.38 ms | 1.75 ms | 7.10 ms |
| Online P95 | 14.42 ms | 11.24 ms | 10.55 ms |
| Online P99 | 14.82 ms | 11.45 ms | 16.21 ms |

The guarded path retains a lower P95 than FP32, with the measured
P99 increased by full scans after the guarded path. The online
median among fallback instances is 15.36 ms. This demonstrates
an exact warm local response under the tested inputs and arithmetic,
plus a latency trade-off at the tail. It does not establish service
tail latency with queueing or a cold backing file.

[Aggregate results](results.json)

## Reproduce

Use the original CoF archive and checkpoint with Slurm CUDA and an
AVX-512 VNNI host. The driver compiles a C++17 shared library.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.real_title_guard QWEN_EMBEDDING NEW_TEMP_LIBRARY \
  NEW_RESULTS benchmarks/real-title-runtime-v1/results.json
```
