# Compare topic-label scope while keeping recommendations fixed

The [major-descriptor experiment](../elife-mesh-v1) produced only
ten requests. Use the same 322 sources and [NLM MeSH](https://www.nlm.nih.gov/mesh/meshhome.html)
snapshot to build a larger panel that includes ordinary and major
assignments. Keep descriptors present on 8–40 articles, with the
same routine-population exclusions as the earlier study. For each
descriptor, apply the original deterministic selection of two role
pairs with positive but incomplete support. Verify that both roles
in each cited source belong to one author using the publisher
declarations.

Run topical top four, topical top eight, unrestricted same-person
nomination, top-eight-constrained same-person nomination and the
[PLOS-calibrated similarity floor](../role-score-gate-v1) unchanged.
Use the pinned title/abstract retrieval index, verifier cutoff and
models. Neither type of MeSH occurrence is an article feature used
by search. Compute metrics twice on the same ordered recommendations.
The broad count treats any exact descriptor assignment as a hit.
The focused count requires the same descriptor to be marked major.
Keep every query in both denominators, including requests with no
major reference satisfying the role pair, and report that reachability
explicitly. Also report the subset of descriptors whose exact topic
strings never occur in the original publisher heading or keyword
metadata. This identifies a lexical transfer slice, with no claim
that it introduces completely different concepts or articles.

For both label scopes, absence is unresolved. The source-level
assignments do not establish whether the returned title and abstract
support the scope of a request. The larger panel gives a better
comparison of the proxy metrics, with its 322-document corpus size
and indexing conventions still fixed. Timing includes online
encoding, retrieval and verification from the reused experimental
driver. It excludes remote service and queue overhead.

## Results

RTX 5090 jobs `43802` and `43804` reproduce all non-timing metrics.
The panel has 110 queries across 55 descriptors. At least one article
with a major assignment and both roles credited to the same person
exists for 31 queries. Forty-one descriptors (82 queries) have no
exact counterpart in the eLife keyword or heading metadata.

| Policy, 280 declarations indexed | Issued | Any matching descriptor | Major descriptor | Reads |
| --- | ---: | ---: | ---: | ---: |
| Topic four | 225 | 142 | 19 | 440 |
| Topic eight | 260 | 165 | 17 | 880 |
| Person four | 255 | 154 | 18 | 440 |
| Top-eight person four | 246 | 150 | 17 | 364 |
| PLOS-calibrated topical floor | 248 | 148 | 20 | 440 |

Compared with topic four, person four gains 12 broad descriptor
slots (+2.18 percentage points of P@5, descriptive paired 95%
interval +0.91 to +3.82), and loses one major slot (−0.18 points,
interval −1.09 to +0.55). The fraction of issued recommendations
with any matching descriptor decreases from 142/225 (63.1%) to
154/255 (60.4%). Thus the additional broad hits coexist with 18 more
off-label citations. The exact-label transfer subset gains nine
broad slots and one major slot. These outcomes show why broader
coverage and topical focus need separate measurements.

With the sparse mask of 144 declarations, person four gains one
broad slot with no major change; the PLOS-calibrated floor gains
one major slot with no broad change. At full available coverage,
top-eight person nomination requires 364 rather than 880 source
reads compared with topic eight. Job 43802 measures local P95 of
26.45 versus 45.76 ms, but the route misses 15 broad tagged slots
and fails the existing quality noninferiority gate. Neither label
scope establishes evidence precision against human judgments.

[Aggregate replay](results.json)

## Reproduce

Rebuild the pinned [eLife text projection](../elife-role-nomination-v1)
and use the [released MEDLINE snapshot](../elife-mesh-v1). Run on one
Slurm CUDA allocation with the frozen open checkpoints.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.elife_mesh_scope ELIFE_PROJECTION \
  benchmarks/elife-mesh-v1/mesh.json benchmarks/elife-mesh-v1/results.json \
  benchmarks/role-score-gate-v1/results.json QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
