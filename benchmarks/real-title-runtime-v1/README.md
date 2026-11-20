# Test native INT8 title retrieval on real SciRepEval publications

The [native INT8 study](../title-int8-runtime-v1) measured scale
using repeated title vectors. Here the index contains actual
publications from the CoF SciRepEval reviewer profiles. Rebuild the
[pinned CoF archive](https://github.com/yuz9yuz/CoF) with the existing
loader, keeping the source-ID inventory and the 91-query filtered
cohort. Exclude queries whose IDs, normalized titles or full texts
occur in the source collection. Keep title collisions between distinct
source IDs and count distinct normalized title strings separately.

Use the same frozen Qwen3-Embedding-0.6B checkpoint for both paths.
Encode source titles offline with a 128-token cap and batch size 128.
For a live request, encode the supplied query title and abstract
with the full 512-token cap. Compare bounded FP32 dot-product
selection with the native INT8 VNNI top-two kernel. Use all sources
as eligible since this dataset does not supply same-person role
postings. Break score ties by source ID. Both methods query the
same warm SQLite database and serialize the selected source ID,
its title, an abstract excerpt and associated reviewer IDs.

Run a cached-vector mode and a live-encoding mode. Alternate methods
and replay each query three times. Report ordered serialized response
agreement, selected source-set agreement and first-source agreement
separately. Keep the original sparse reviewer relevance labels as a
secondary proxy, recording positive and judged coverage among the
reviewers of the two returned sources. An unjudged reviewer is not
a confirmed negative. These labels do not adjudicate the abstract
excerpt or certify any individual contribution.

The measured path starts at query input or at its cached vector and
ends at local materialized citation bytes. It does not add a semantic
verifier, a second retrieval view, attribution reasoning, network
or service queues. Record title encoding cost and persistent index
payload separately. No third-party paper text enters the release.

## Results

CoF provides 73,257 publication IDs with 72,272 distinct normalized
titles. One FP32 title matrix occupies 300,060,672 bytes, compared
with 75,601,224 bytes for INT8 with scales and row sums. Title
encoding takes 24.4 s in the first run on ws5's RTX 5090.

| Warm local P95, job 43848 | Bounded FP32 | Native INT8 VNNI |
| --- | ---: | ---: |
| Cached vector to serialized citations | 6.22 ms | 1.74 ms |
| Online query to serialized citations | 14.09 ms | 10.80 ms |

All three jobs (`43848`, `43850`, `43852`) agree on candidate and label
counts. The last run also checks response bytes for every query and
method across modes and repeats, confirming exactly two distinct
queries with a changed ordered response. FP32 and INT8 agree on
the first citation and unordered source
set in 270 of 273 paired requests. They return byte-identical
ordered responses in 267 of 273. Both methods reach at least one
known positive reviewer on 40 of 91 queries under soft labels
and 29 of 91 under hard labels. For each rubric, only 86 of 206
returned query–reviewer pairs are judged. Equal aggregate counts
do not establish equal relevance or evidence support.

These results show the storage and local latency advantage on a
real title index, with a small but observed difference in citation
selection. They support evaluating a boundary guard or an explicit
approximation policy before claiming exact output at scale.

[Aggregate replay](results.json)

## Reproduce

Use a Slurm GPU allocation on an x86 machine with AVX-512 VNNI and
a C++17 compiler. The existing loader can use the pinned archive
under `artifacts/public/cof/data.zip`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.real_title_runtime QWEN_EMBEDDING NEW_TEMP_LIBRARY NEW_RESULTS
```
