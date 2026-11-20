# Compress the title view and measure the effect on cited experts

The [dual-view role experiment](../role-field-ranking-v1) found more
major-topic references but doubled dense vector storage. Leave the
main hybrid ranking, role completeness and verifier untouched.
Change only the ranking of the additional title view. Reuse a single
query vector for both views and the same two-plus-two nomination rule.
No quantization parameters are fitted on queries or relevance labels.

| Format | Title score |
| --- | --- |
| FP32 | Original normalized-vector dot product |
| INT8 | Symmetric per-vector scaling, signed 8-bit codes, int32 accumulation and scale restoration |
| Sign bit | Packed component signs, ranked by Hamming distance |

Use the same 46 PLOS requests and 110 eLife MeSH requests under both
declaration masks. Require the original FP32 configuration to reproduce
all earlier issuance and tag counts. Report ordered candidate agreement
and ordered expert-and-citation agreement for every compressed arm.
Run the live verifier on every selected batch. Topic and major-topic
labels are used only after ranking. The MeSH assignments do not
establish human-judged relevance.

Report persistent title vector payload, including stored INT8 scales,
separately from process memory, build overhead and NumPy temporaries.
For example, the generic integer matmul may materialize int32 operands
while scoring. Timings measure title score and sort separately and
include complete warm local request execution. The original hybrid
ranking cost is shared for comparison. These tests do not claim memory
bandwidth or service speed at production scale.

## Results

RTX 5090 jobs `43825` and `43827` match all non-timing metrics.

| Catalog and declaration masks | Ordered candidates equal, INT8 / sign bit | Ordered expert–citation lists equal, INT8 / sign bit |
| --- | ---: | ---: |
| PLOS, 46 requests and two masks | 92 / 6 of 92 | 92 / 79 of 92 |
| eLife, 110 requests and two masks | 220 / 47 of 220 | 220 / 188 of 220 |
| Combined | 312 / 53 of 312 | 312 / 267 of 312 |

For 322 eLife titles, the FP32, INT8 and sign-bit payloads are
1,318,912, 331,016 and 41,216 bytes. Keeping the original hybrid
index unchanged and replacing the FP32 title matrix with INT8
reduces its added dense payload from 100% to 25.1% of the
original matrix. In these replays the INT8 route returns all
FP32 candidates, experts and citations in exactly the same order.
At full available coverage on eLife, the sign-bit route preserves
12 requests with a major witness, compared with 13 for FP32
and INT8. It produces 24 major-tagged slots instead of 27.
Output equality is an observed property of these cohorts.

The current NumPy implementation is a storage optimization.
On the full eLife mask in job 43825, title scoring and sorting
P95 rises from 51 µs in FP32 to 197 µs in INT8. Warm request P95
moves from 29.33 to 29.45 ms. The measured payload and short
request comparison do not justify a throughput claim. Further
work on integer kernels or large collections should separately
check score ordering, temporary allocation and service load.

[Aggregate results](results.json)

## Reproduce

Rebuild the [original PLOS projection](../plos-topic-role-holdout-v1)
and [eLife projection](../elife-role-nomination-v1), then use the pinned
checkpoints in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.title_index_compression PLOS_HOLDOUT \
  benchmarks/plos-topic-role-holdout-v1/manifest.json ELIFE_PROJECTION \
  benchmarks/elife-mesh-v1/mesh.json benchmarks/role-field-ranking-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
