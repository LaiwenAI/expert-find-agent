# Fetch only the role postings needed to fill the evidence budget

Evaluate query-time residency separately from the offline placement
of declarations. The [fixed history index](../plos-fixed-role-index-v1)
could not reliably improve recommendation quality on novel topics.
Here the same compiled same-person relation can live outside the query
process. Each policy sees the same completeness bitmap. A known
article is eligible if the requested two roles occur on the same
author. Articles with unavailable declarations stay eligible. This
distinguishes a missing positive posting from a known negative.

Return the first four eligible articles in a fixed topical order.
Compare an in-process posting set, a SQLite range lookup for the
entire requested role-pair list, and SQLite point lookups in rank
waves of four or eight. At each wave, query only IDs with known
declarations. Stop after the fourth eligible article or after
exhausting the ranking. The SQLite index uses a composite primary
key of role pair and article ID. All four policies use identical
ranking, candidate budget, verification and citation formatting.
The replay asserts equality of rank lists, ordered candidates, raw
reranker scores and expert–citation outputs for every paired request.

The real-catalog replay uses the 46 requests and two published
completeness masks from the [PLOS nomination replay](../plos-bounded-topic-v1).
It uses online query encoding, retrieval, metadata lookup, topical
reranking and result assembly in its warm request timing. It excludes
process startup, network, contention, online parsing and storage
queueing. Index construction is offline; complete declarations and
topic-index data are assumed available for compilation.

The separate lookup-only stress test keeps a deterministic permutation
of the original 78 candidates at the head and appends replicated
records with distinct IDs up to 39,936 sources. It preserves role
frequencies but creates correlated copies. It does not run a model or
evaluate expert quality, corpus-scale embedding latency, or real
document growth. Compare warmed SQLite steps with the resident index
so the footprint reduction can be considered alongside latency.

## Results

Slurm job `43757` used an RTX 5090 and replayed five repetitions per
query under each completeness mask. All 460 request instances matched
exactly across the four lookup policies, including ordered citations.
Each request verified four sources. For the 46 requests with all 78
declarations available, every policy returned 52 tagged and 16 off-label
recommendations. With 42 declarations available, the counts were 41 and
7. The timing of the index operations includes tracing counters.

| Real requests, 78 declarations | Metadata lookup P95 | Warm request P95 |
| --- | ---: | ---: |
| Resident | 17.71 µs | 36.28 ms |
| SQLite complete posting list | 43.96 µs | 36.30 ms |
| SQLite waves of four | 85.49 µs | 36.34 ms |
| SQLite waves of eight | 69.46 µs | 36.34 ms |

Slurm CPU job `43759` ran 189 requests per policy and mask at each
synthetic size. With all declarations known, eight-ID waves had the
following lookup-only profile.

| Sources | Resident P95 | Complete-list SQLite P95 | Eight-ID SQLite waves P95 |
| --- | ---: | ---: | ---: |
| 78 | 4.74 µs | 12.23 µs | 25.01 µs |
| 4,992 | 5.09 µs | 448.25 µs | 28.64 µs |
| 39,936 | 6.33 µs | 4,019.77 µs | 32.19 µs |

The largest SQLite file was 77,467,648 bytes, with 1,423,872 posting
rows. The complete-list path returned 2,834,944 rows across the 189
requests at that size. Eight-ID waves returned 959 rows with 378 calls.
A fixed ranking head makes this chiefly a test of avoiding a large
irrelevant posting tail. In-memory postings remain faster. Neither the
lookup tail result nor the close real-catalog timings demonstrate an
end-to-end latency reduction on a large catalog. The useful property is
exact candidate selection with bounded posting retrieval when the role
index is external to the process.

[Real replay](live.json) · [Lookup stress test](scale.json)

## Reproduce

Use the [pinned projection](../plos-topic-role-holdout-v1) and original
open checkpoints in a Slurm CUDA allocation for the real replay.
The lookup stress test uses a Slurm CPU allocation.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.lazy_role_index HOLDOUT_PROJECTION \
  benchmarks/plos-topic-role-holdout-v1/manifest.json QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/plos-bounded-topic-v1/results.json NEW_LIVE_RESULTS --repeats 5
python -m experiments.credit.lazy_role_index_scale HOLDOUT_PROJECTION \
  benchmarks/plos-topic-role-holdout-v1/manifest.json NEW_SCALE_RESULTS
```
