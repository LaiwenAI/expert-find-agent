# Resident evidence bundles for paper-centric reviewer ranking

Use the 148 paper-centric comparisons from the
[matched MERIT release](../merit-author-retriever-v1).
Group by the 110 target papers, verify every supplied
reviewer has one stable publication history, and score
all displayed candidates for a target in one request.
The workload contains 193 distinct reviewers and at most
five supplied candidates per target.

Use the same MERIT 8B adapter final score in every route.
The live control precomputes the backbone's publication
vectors. At request time it runs the author prefilter to
select up to three relevant publications per candidate
and encodes their query-dependent profile texts. Its
target paper is embedded only once inside each model stage.

All resident routes precompute the complete profile
vectors and hold them in memory. Each request encodes
the target once. The fixed-prefix route uses the first
three released publications. The one-bundle coverage
route uses frozen 0.6B publication vectors and greedily
chooses up to three representative sources. At each step,
maximize the total similarity improvement from representing
each publication by its closest chosen source. The
multi-bundle route uses each selected representative as
a center and adds its two nearest profile publications.
Score a reviewer by the largest cosine across its resident
bundles. Every bundle has at most three original sources.

One warm request per route precedes measured requests.
Rotate route order and measure target encoding, scoring and
live profile preparation from query input to candidate
scores. Pair latency resampling by target and pairwise
quality resampling by anchor. Count distinct benchmark
requests, model input texts, and stored vector bytes
without Python object or model weight overhead. The
offline build time includes the shared implementations
and does not describe individual per-route build speed.
This is a supplied-candidate ranking experiment, with no
catalog search, transport, or independent citation review.

## Results

Slurm job `45243` ran on one RTX 5090. The live control
reproduced the aggregate paper-centric metrics from the
matched MERIT benchmark. No offline input was truncated.

| Route | Correct / 148 | Weighted loss | Online texts / 110 requests | Warm P95 | Stored FP32 vectors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Author-conditioned live | 112 | 0.1660 | 455 | 575.82 ms | 14.01 MB |
| Fixed first three | 111 | 0.1966 | 110 | 50.28 ms | 3.16 MB |
| One coverage bundle | 108 | 0.2080 | 110 | 50.25 ms | 3.16 MB |
| Multiple coverage bundles | 113 | 0.1660 | 110 | 50.23 ms | 7.19 MB |

Multiple coverage bundles reduce the warm P95 by 91.3%.
The paired 95% interval for the P95 difference is -656.75
to -467.68 ms. Relative to the live route, five comparisons
become correct and four become incorrect. The paired accuracy
difference is +0.68 percentage points (95% anchor interval
-3.45 to +4.61); noninferiority is unproven. The multiple
bundle route stores 439 vectors for 193 reviewers. The
shared offline build encodes 663 distinct profile texts in
75.26 seconds after coverage grouping. This study establishes
a latency advantage for ranking a supplied pool on one GPU
under warm serial execution. Full-catalog retrieval,
concurrent serving and candidate evidence correctness need
separate measurements. [Aggregate results](results.json).

## Reproduce

Follow the pinned inputs and GPU setup from the
[author retriever experiment](../merit-author-retriever-v1).

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.merit_resident_profile_bundles \
  MERIT_TEST_DIR MERIT_SCRIPT_DIR RATE_DIR   QWEN8B_CHECKPOINT MERIT_ADAPTER QWEN06B_CHECKPOINT NEW_RESULTS
```
