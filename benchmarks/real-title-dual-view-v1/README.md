# Test dual-view citation retrieval under a two-source budget

The [single-view replay](../real-title-block-recheck-v1) verified
local acceleration for title search. This experiment asks whether a
second view pays for its additional scan when every method returns
exactly two distinct sources. Reuse the leakage-filtered CoF
SciRepEval cohort of 91 queries, 73,257 publication IDs, and the
frozen Qwen3-Embedding-0.6B checkpoint.

Encode each source title alone with a 128-token cap or its title and
abstract with a 512-token cap. Use the same 512-token query encoding
in all arms. Compare the top two titles, the top two title-abstract
sources, and a dual view. The dual view returns the highest scoring
title-abstract source followed by the highest scoring title source
that differs from it. Compare bounded FP32 with the guarded INT8
title scan plus aligned 32-row FP32 rechecks inside the dual view.
The title-abstract scan uses FP32 for both dual arms.

Replay each query three times in rotated method order in both cached
and online query-encoding modes. Measure through warm local citation
serialization. Check the guarded title band against full FP32 and
require byte-identical output for both dual variants on every replay.
Record both view-build times and the separate exact title backing
payload.

Use the original soft and hard reviewer labels to count queries
whose two sources reach at least one known positive reviewer. Compute
paired query bootstrap percentile intervals for the dual-view
difference relative to title-abstract retrieval. Record judged and
total returned query–reviewer pairs so label sparsity is visible.
These are reviewer-familiarity proxies. This corpus provides no
same-person role postings, attribution judgments, or evidence for
the suitability of specific excerpts. Timing covers warm local
retrieval through citation serialization and excludes semantic
verification, storage cold starts, and service queues.

## Results

Slurm job `43915` ran on ws5 with an RTX 5090 and the validated
contiguous OpenBLAS recheck. The guard matched all 1,323 in-band
FP32 scores and all 273 serialized dual-view responses in each mode.

| Two-source route | Soft positive queries | Hard positive queries | Online P95 | Online P99 |
| --- | ---: | ---: | ---: | ---: |
| Title only, FP32 | 40 / 91 | 29 / 91 | 15.39 ms | 15.58 ms |
| Title and abstract, FP32 | 52 / 91 | 32 / 91 | 15.33 ms | 15.69 ms |
| Dual view, FP32 | 46 / 91 | 31 / 91 | 21.12 ms | 21.88 ms |
| Dual view, guarded title scan | 46 / 91 | 31 / 91 | 17.24 ms | 17.53 ms |

The dual view changed the source set for 77 queries. Under soft
labels, the fixed second-view slot gained 2 positive queries and lost
8, a paired difference of −6.6 percentage points with a 95% bootstrap
interval of [−13.2, 0.0]. Under hard labels it gained 3 and lost 4,
a difference of −1.1 points with an interval of [−6.6, +4.4].
Only 102 of 217 dual-view query–reviewer pairs and 107 of 203
title-abstract pairs were judged under either rubric.

The guarded scan saved 3.88 ms at online P95 within the same dual
policy. The resulting path still took 1.91 ms longer at P95 than the
single title-abstract view. The two FP32 matrices each occupied
300,060,672 bytes. The guarded title path also needed 75,894,252
bytes of compact scan data and retained the title FP32 backing.
Offline encoding took 24.5 s for titles and 299.6 s for titles with
abstracts. The result supports the accelerated implementation as
an ablation and does not support reserving a title slot by default
on this cohort.

[Aggregate replay](results.json)

## Reproduce

Run with CUDA through Slurm on the validated AVX-512 VNNI host. Pass
an existing scratch directory on the node's storage, a fresh result
JSON, and a completed replay JSON from the prior contiguous-block
study. Keep BLAS configured as in that study.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.real_title_dual_view QWEN_EMBEDDING SCRATCH NEW_RESULTS PRIOR_RUN
```
