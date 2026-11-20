# Add a title view to role-compatible first-query nomination

The [stage diagnostic](../elife-mesh-stage-v1) found that most
reachable major-topic references were never read. Test whether a
second retrieval view can nominate some of these references before
verification. Use the same embedding model to encode titles offline
alongside the existing title-and-abstract index. Reuse one query
vector for the original hybrid ranking and the title-only dense
ranking. The title view cannot access publisher topics, keywords,
MEDLINE labels or test outcomes. Neither weight nor budget allocation
is learned or selected from the results.

Compare the following at four reads per request, using topical eight
as the deeper control. A role-eligible source either credits the
requested pair to the same person or has an incomplete declaration.

| Policy | Candidate selection |
| --- | --- |
| Topic four | First four hybrid candidates |
| Person four | First four role-eligible hybrid candidates |
| Mixed views without role screening | Two hybrid candidates, then two unused title candidates |
| Mixed views with role screening | Two role-eligible hybrid candidates, then two unused role-eligible title candidates |
| Title with role screening | First four role-eligible title candidates |

Deduplicate cross-view nominations. Refill an insufficient list from
the original hybrid rank. The original topic reranker reads complete
title/abstract views for all policies. It uses the same cutoff and
declared same-person role binding before recommendation.
The stage diagnosis counts the 31 independently reachable eLife
queries that reach the read list, pass the verifier and return a
major-topic citation. The PLOS panel only provides topic tag counts.
Compare paired outcomes and off-label issuance under both masks.
A publisher or MEDLINE tag is an article-level proxy, not a human
assessment of the recommender's quoted evidence.

Run both frozen public catalogs and recheck the existing baselines
before judging any new variant. The timings include online encoding,
rank, selection and verification under warm serial execution. The
title matrix size and offline title encoding time are separate.
Service queueing and production-scale memory effects are outside the
replay.

## Results

RTX 5090 jobs `43818` and `43820` match all non-timing metrics.
The published controls reproduce on the 46-query PLOS and
110-query eLife panels. Results below use the full available
declaration masks. Counts are recommendation slots except for
the final column, which counts queries.

| Policy | PLOS tagged / off-label | eLife broad / off-label | eLife major slots | eLife queries with a major witness |
| --- | ---: | ---: | ---: | ---: |
| Topic four | 37 / 4 | 142 / 83 | 19 | 9 |
| Topic eight | 49 / 14 | 165 / 95 | 17 | 9 |
| Person four | 52 / 16 | 154 / 101 | 18 | 10 |
| Mixed views without roles | 34 / 0 | 136 / 68 | 26 | 12 |
| Mixed views with roles | 46 / 11 | 152 / 86 | 27 | 13 |
| Title with roles | 43 / 10 | 133 / 59 | 30 | 12 |

On the 31 eLife requests with a known same-person major target,
adding title retrieval and role screening changes the stage counts
from 11 selected, 11 passing and 9 issued with a major citation
to 16, 14 and 13. The final change is five gains and one loss.
The major-score difference is +8 slots, or +1.45 points of P@5
(descriptive topic bootstrap interval +0.18 to +3.09).
With the sparse declaration mask, this variant issues a major
witness on 12 queries. It adds 16 broad slots relative to
the mixed-view route without roles under full coverage, while
the broad total is two slots below the person-only route.
In PLOS, the mixed role route finds six fewer tagged slots and
five fewer off-label slots than person four.

The mixed role route still fails the broader topic-label
noninferiority test versus topic eight. In job 43818, eLife
topic four, person four and the mixed role route have local P95
of 26.58, 29.44 and 29.47 ms, respectively. The additional dense
matrix takes about 1.32 MB for 322 records, matching the size of
the original matrix. Offline title encoding takes 0.145 s in
that run. This improves coverage of an independently indexed
major-topic signal, with extra index memory and gains concentrated
in this panel. Human support and large-catalog efficiency remain
to be tested.

[Aggregate results](results.json)

## Reproduce

Use the original [PLOS holdout](../plos-topic-role-holdout-v1) and
[eLife](../elife-role-nomination-v1) projections and pinned checkpoints
in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.role_field_ranking PLOS_HOLDOUT \
  benchmarks/plos-topic-role-holdout-v1/manifest.json benchmarks/plos-bounded-topic-v1/results.json \
  ELIFE_PROJECTION benchmarks/elife-mesh-v1/mesh.json \
  benchmarks/elife-mesh-scope-v1/results.json QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
