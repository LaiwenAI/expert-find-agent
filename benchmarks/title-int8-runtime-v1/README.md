# Fuse INT8 title scoring, role filtering and top-k selection

The [title compression replay](../title-index-compression-v1)
preserved the two-view shortlist but its NumPy integer path was
slower than FP32. It can promote full matrices to int32 during
dot-product evaluation and then sorts the whole corpus. Compare a
single native scan that skips ineligible rows and retains only the
remaining title slots. Keep the first two eligible hybrid candidates,
exclude them from the title scan, and use source ID to break ties
identically to the stable reference sort.

The native kernel works on signed 8-bit title storage. On AVX-512
VNNI hardware, shift each query component by 128 to use the unsigned
query operand and subtract 128 times the precomputed signed row
sum. This exactly recovers the signed INT8 dot product. Keep a
portable scalar implementation of the same reduction. Include
per-vector scale restoration and candidate eligibility in the
timed path. No full int32 title matrix or full score sort is needed
by the native route.

Two measurements share an encoder and title vectors. The first
uses all 110 frozen eLife queries under two declaration masks.
It times candidate selection after the hybrid ranking and query
vector are available. Every native shortlist must match the
ordered NumPy INT8 reference, and FP32 agreement must reproduce
the previously reported counts. No GPU reranker is run because
the experiment checks identical inputs to that stage.

The second measurement repeats the eLife title vectors up to
41,216 rows to stress the score-and-sort operation alone. It
measures the top two with all rows eligible. These are synthetic
sizes with deliberately repeated rows, so this checks kernel
cost and tie handling. It does not validate retrieval quality
or end-to-end latency on a larger corpus. The strong FP32 comparison scores
with OpenBLAS and takes the bounded number of maxima without a full
sort. All paths use warm local execution with one OpenBLAS thread. Report persistent
payload and online latency separately.

## Results

The CPU is an AMD Ryzen 9 9950X3D on Slurm node ws5, with AVX-512
VNNI available. GCC 13.3 compiled the local shared library; the
embedding stage runs on an RTX 5090. Jobs `43837` and `43839` match
all correctness and payload counts. In job 43837, all 220 real
shortlists match the original INT8 and FP32 orders. The bounded
FP32 implementation also matches the stable FP32 sort on those
requests and the synthetic top-two comparisons.

| Timing P95, job 43837 | FP32 sort | FP32 bounded | NumPy INT8 sort | Native scalar | Native VNNI |
| --- | ---: | ---: | ---: | ---: | ---: |
| eLife candidate selection, 322 sources | 33.1 µs | 40.5 µs | 167.1 µs | 75.7 µs | 57.9 µs |
| Synthetic top two, 5,152 rows | 369.6 µs | 217.7 µs | 2,149.7 µs | 478.8 µs | 84.8 µs |
| Synthetic top two, 41,216 rows | 6.24 ms | 5.35 ms | 25.60 ms | 3.85 ms | 1.18 ms |

In the largest synthetic case VNNI is about 4.5 times faster than
bounded FP32. Synthetic data repeat the same title vectors and can
hide corpus-distribution effects. On original eLife requests, the
fastest FP32 candidate path remains faster at P95. The native
persistent title payload is 332,304 bytes at that size, about 25.2%
of the FP32 payload, including per-title float scales and int32 row
sums. The real selection comparison omits query embedding, hybrid
retrieval, verification, queues and service I/O. No end-to-end
acceleration or larger-corpus quality improvement is established.

[Aggregate timing and equivalence checks](results.json)

## Reproduce

Rebuild the pinned [eLife projection](../elife-role-nomination-v1)
and use the frozen [MeSH snapshot](../elife-mesh-v1). The Python
driver compiles the included C++ source with the local compiler.
It needs Slurm CUDA for encoding. VNNI executes only on
supported x86 hosts with a compatible vector width.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.title_int8_runtime ELIFE_PROJECTION \
  benchmarks/elife-mesh-v1/mesh.json benchmarks/title-index-compression-v1/results.json \
  QWEN_EMBEDDING NEW_TEMPORARY_LIBRARY NEW_RESULTS
```
