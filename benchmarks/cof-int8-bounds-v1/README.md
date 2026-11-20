# INT8 filtering with exact expert and source refinement

Run on the frozen CoF KDD and SciRepEval full-catalog
panels from the [profile pooling study](../cof-global-profile-pooling-v1).
Compare dense FP32 dot products with a rowwise symmetric
INT8 index. Bound the score error using the L2 norms of
the quantization residuals and a FP32 accumulation allowance.
Retain an owner if its upper score bound reaches the tenth
highest owner lower bound. Within each retained owner, retain
sources whose upper bound reaches the third highest lower
bound, or the last lower bound for profiles with fewer
sources. Recompute these source scores in FP32. Equal
bounds stay eligible.

All timed requests encode a query live with the same frozen
0.6B model. The run alternates route order after warming.
Compare the exact owner order and the ordered evidence IDs
before verification. There is no evidence model, transport,
or service queue in the timer.

## Results

Slurm array 45286 ran both panels on an RTX 5090.

| Panel | Identical owner and source output | FP32 rows per request, dense / bounded | Retrieval P95, dense / bounded | Query and retrieval P95, dense / bounded |
| --- | ---: | ---: | ---: | ---: |
| KDD | 122 / 122 | 99,594 / 184.8 | 1.77 / 2.19 ms | 10.98 / 11.36 ms |
| SciRepEval | 59 / 59 | 73,257 / 182.2 | 1.23 / 1.76 ms | 10.58 / 11.00 ms |

The route keeps over 99.7% of source rows out of the
FP32 refinement, yet its integer scan, host transfer and
bound reductions increase observed tail latency. The
implementation still keeps the full FP32 vectors resident,
plus about 102 MB of INT8 codes for KDD or 75 MB for
SciRepEval. The codes are additional resident storage.
The number of pruned FP32 dots is not a runtime or memory
saving in this implementation.

[Full KDD aggregates](kdd.json) · [Full SciRepEval aggregates](scirepeval.json)

## Reproduce

Use the cached source vectors and the pinned 0.6B checkpoint
from the [full-inventory retrieval benchmark](../cof-global-profile-pooling-v1).
Run the following inside a Slurm CUDA allocation with a fresh
output path. Pass `SciRepEval` for the second panel.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_int8_bounded_retrieval \
  KDD QWEN06B_CHECKPOINT NEW_RESULTS
```
