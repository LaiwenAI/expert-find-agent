# Defer citation sorting in exact enterprise hybrid search

The [public resident timing](../resident-directed-source-timing-v1)
found exact-rank vectorization reduced warm client P95. The
enterprise hybrid route already uses resident NumPy BM25
postings. This run tests the remaining owner-aggregation work
on the [frozen hybrid baseline](../enterprise-hybrid-rescue-v1),
with 80 development requests and 52,667 source records.

## Exact ranking change

Both paths score the same source rows with the same BM25 arrays
and exact dense index. They use reciprocal rank fusion with the
same 50-owner depth, five final experts, and two sources per
expert. The original path materializes all lexical results as
a dictionary and sorts every matching expert's sources. The
deferred path reduces the active lexical source scores to a
maximum per owner and sorts the owners. After owner fusion, it
sorts lexical sources for only the final experts, including
dense-only owners with lexical evidence below the lexical owner
cutoff. An offline source-ID-ordered posting list preserves
the original tie-break when source scores tie. The dense path
and fusion algorithm are shared.

Measure the exact same qualified search contract per route
with fresh qualification parsing, GPU query encoding, hybrid
retrieval, proof verification, and ordered citations. Check
for exact equality of qualification spans, retrieved IDs and
fusion metadata, citation ranges, acceptance and unresolved
state. The experiment uses the hybrid configuration that stops
on first-round abstention. It builds no alternate qualification
or second-round decision method.

Separately run paired retrieval-only measurements with a shared
query vector. Interleave arm order and direction, taking the
median of six warm runs for each query and arm. Whole-query
timing uses one paired run per route and request, with warm
models and indices. Report descriptive paired request bootstrap
intervals for P95 differences. Offline index creation and HTTP
are outside the timed request, and this is a single-worker
development replay. Only aggregate counts, settings and digests
are published. No proprietary content, person identities, or
query traces are included.

## Results

| Measurement | Legacy P50 (ms) | Deferred P50 (ms) | Legacy P95 (ms) | Deferred P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Retrieval with shared query vector | 16.02–16.34 | 3.59–3.64 | 22.98–24.22 | 6.24–6.31 |
| Retrieval during complete request | 18.50–19.03 | 3.94–4.02 | 33.30–33.88 | 7.14–7.15 |
| Complete request | 285.81–287.70 | 264.26–265.86 | 1,649.82–1,672.86 | 1,635.47–1,657.80 |

Ranges span two RTX 4090 jobs. During complete search, retrieval
P95 falls by about 79% in both runs. Complete-request P95 falls by
14.36 and 15.05 ms (approximately 0.9%). The paired descriptive
P95-difference intervals are [−22.82, −2.75] and [−20.66, −7.34] ms.
The unchanged qualification parser takes 1.29–1.31 seconds at P95,
which limits the impact of faster ranking on this workload.

All 80 qualification, candidate, and citation payloads match across
arms and both traversal orders. Both paths return 55 provisional
slots across 26 requests and perform 1,309 source–criterion checks
per run. The earlier RTX 5090 run archived 51 slots, 25 issued
requests and 1,279 checks. This rerun reproduces the model and input
digests, but those older output counts differ. The paired conclusion
uses only the current execution environment. Model-issued output
equivalence does not measure human-judged quality.

Ownership arrays add 421,336 numeric bytes on top of 61,006,980
BM25 posting bytes, excluding Python object overhead. Building the
new arrays takes about 20 ms after the common lexical build. Both
tasks in Slurm array 47247 on ws6 verified the same full output
signature. [Aggregate timings and exact-output check](results.json).

## Reproduce

On an allocated Slurm GPU, use the same private development
projections and saved exact embeddings from the
[enterprise hybrid benchmark](../enterprise-hybrid-rescue-v1).
Write a fresh aggregate into a private working directory:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_owned_lexical_ranking \
  DEVELOPMENT_QUERY_PROJECTION DEVELOPMENT_CORPUS_PROJECTION \
  SOURCE_VECTORS VECTOR_MANIFEST MODEL_ROOT PRIVATE_SCRATCH \
  benchmarks/enterprise-hybrid-rescue-v1/results.json NEW_AGGREGATE
```

Repeat using `--reverse` and a separate output path.
