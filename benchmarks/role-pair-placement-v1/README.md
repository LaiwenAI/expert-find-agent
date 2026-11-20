# Compact the role index for same-person, two-source requests

Reuse the [frozen PLOS and eLife mixed-publisher corpus](../owner-evidence-bound-v1)
with 452 articles and 211 structured requests. The verifier always reads
two distinct sources and checks both role declarations against the same
checksum-valid ORCID. A match is counted as known when both publisher
topic annotations also match. The corpus-wide topic and embedding
indexes, document identifiers and bylines are shared by all arms.

## Exact projection

A contributor appearing in only one source cannot participate in a
two-source same-person proof. Remove that contributor's role columns
and candidate entries from this proof planner. Keep the source searchable
for topical retrieval and keep its original declaration available to
the verifier. Dropping single-source identities preserves the feasible
pair set under the ranker, including pairs with unresolved role fields.
The rank, cost-independent tie break and topic-only fallback are unchanged.
This applies to the frozen corpus and its identity mapping. A newly linked
source can make an existing contributor eligible, so new or changed
sources require recomputing the affected membership and role payload.
Queries answerable with one source or with unlinked identity evidence
require a separate path.

| Candidate depth per topic | Full role payload | Repeated-author payload | Plan-selection P95, full → projected | Known PLOS / eLife |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 6,114 B | 499 B | 0.165 → 0.083 ms | 68 / 17 |
| 128 | 6,114 B | 499 B | 0.256 → 0.070 ms | 85 / 46 |

The ordered source plan matched on each of the 211 requests at each
depth. The role payload shrank 91.8%, from 441 to 54 sources with
packable roles. These byte counts include each source header and its
packed role bitmaps. They exclude the shared byline and retrieval
indexes, decoded Python data structures, and original declarations. The
projected author order is derived from the existing byline index. The
reported P95 measures plan formation after retrieval and before reading
the declarations, using warm local data. It is a component measurement
and does not establish end-to-end latency or enterprise match quality.
Much of this fixture's sparsity comes from its limited ORCID overlap.

## Equal-byte placement for the original encoding

Compare stable hash ordering, corpus-wide repeated-author degree per
byte, rank-weighted source frequency, and role-witness frequency. The
joint method adds missing members of a pair to cover more training
requests per byte, then fills spare space by witness frequency. It uses
up to eight distinct pairs per training request ordered by retrieval rank
and same-author role fit. This extends the question asked by work on
[static passage pruning](https://eprints.gla.ac.uk/323368/) to a
constraint that depends on two distinct sources.

Five deterministic topic buckets define the split. Each test request is
assigned to the lower bucket of its two topic hashes. Both topics in a
training query must lie outside the held bucket, so at least one topic of
every test query is unseen in training for its fold. All 211 queries
receive exactly one held-out result. The retrieval corpus and its
published role labels are available to all offline placement methods.
Scoring annotations are unused in placement, and held requests do not
drive index updates. This is a within-corpus split with uneven folds and
possible shared contributors. Topic-pair bootstrap intervals in the
results are descriptive.

| Candidate depth | Byte cap | Hash | Identity degree | Query frequency | Witness frequency | Joint coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 917 | 56 | 85 | 70 | 79 | 79 |
| 64 | 1,834 | 54 | 85 | 79 | 84 | 84 |
| 64 | 3,057 | 62 | 85 | 80 | 85 | 85 |
| 128 | 917 | 61 | 131 | 70 | 126 | 126 |
| 128 | 1,834 | 55 | 131 | 79 | 126 | 126 |
| 128 | 3,057 | 86 | 131 | 92 | 127 | 127 |

Counts are known matches out of 211 at the same online two-read budget.
Joint placement and witness frequency achieve identical known counts
and training coverage in this fixture. The corpus-only identity degree
matches the full role payload in known counts at every tested cap,
and the exact projection reproduces full source plans using even fewer
bytes. These results favor the structural projection for this task.
They do not establish a benefit for workload-trained pair placement.

## Reproduce

Restore the same XML snapshots and embedding checkpoint as the
[declaration-read study](../owner-evidence-bound-v1). Slurm jobs
45575 and 45580 on ws3 each used one CUDA GPU and reproduced all
non-timing allocation and quality results. Job 45580 added the exact
projection check. Repeat with depth 128 and order offset 1.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false python -m experiments.credit.role_pair_placement \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json \
  NEW_RESULTS --depth 64 --order-offset 0
```
