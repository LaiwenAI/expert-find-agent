# Use facet-specific views to nominate one challenger

Test whether a compact facet representation improves the second
verification slot on the same 50 expert-judged CSFCube queries.
The [lexical nomination comparison](../csfcube-lexical-challenger-v1)
found more losses than gains before verification. The full-abstract
dense top six contains a strongly relevant paper on substantially
more requests than the top two. This experiment tests whether a
second representation can expose that evidence without verifying
more papers.

Both original judged pools and pools with all grade 2–3 papers
withheld are unchanged. Every route ranks the eligible pool by the
same full-abstract embedding and preserves the top dense paper.
The facet is provided by the benchmark. For compact views, keep the
title and up to three abstract sentences selected by the release's
predicted rhetorical labels. Fall back to the first abstract
sentence if no matching label occurs. The matched-lead control
keeps the same number of sentences at the beginning on that query
or candidate. Matching sentence counts does not match token counts.

| Route | Remaining papers given to the verifier |
| --- | --- |
| Dense two | Second paper from full-abstract dense ranking |
| Matched lead on both sides | Best of dense ranks 2–6 by compact lead-to-lead similarity |
| Facet query | Best of dense ranks 2–6 by facet query to full-abstract similarity |
| Facet on both sides | Best of dense ranks 2–6 by compact facet-to-facet similarity |
| Batched facet on both sides | Same facet rule, with the full and compact query encoded in one forward |
| Dense six | All five remaining dense candidates |

All routes use the frozen 0.6B Qwen embedding and reranker from the
previous comparison. The reranker reads the full original query
with its facet instruction and each full candidate abstract, in
one batch. Fit the same score-and-margin profiles out of fold at
the 5%, 10%, and 20% calibration ceilings. Keep all facets of a
query paper on the same side of each split. Report strong exposure
before reranking, strong winners before abstention, and accepted
grades. Route timings add measured full dense search, compact query
encoding and scoring when applicable, and verification. The batched
route measures one combined query forward and checks shortlist and reranker score agreement against the separate
forwards. Candidate-side compact vectors are precomputed and
deduplicated. Report offline work separately. Service queues
are outside the serial P95.

Facet representations have prior art in [ASPIRE](https://aclanthology.org/2022.naacl-main.331/),
which matches sentence-level facets learned from co-citation
supervision. This diagnostic uses frozen general-purpose embeddings
and tests slot allocation. It does not compare against the ASPIRE
checkpoint or claim novelty for facet-specific embeddings.

## Results

Slurm jobs 43331 and 43333 completed on one RTX 5090 on ws5.
The second job reproduces the earlier discrete outcomes and adds
the joint encoding arm. The facet-on-both-sides route improves
strong candidate exposure from 35 to 37 of 50, gaining four
requests and losing two. The matched-lead control exposes 34.
Dense six exposes 45. Verifying both facet candidates puts a
strong candidate first on 28 requests before thresholding,
versus 29 with dense two.

| Policy | Strong returns / 50 at 5% | Withheld grade 1 / 0 at 5% | Strong returns / 50 at 10% | Withheld grade 1 / 0 at 10% | Present P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense two | 4 | 3 / 1 | 13 | 4 / 2 | 26.0 ms |
| Matched lead on both sides | 7 | 2 / 2 | 13 | 5 / 2 | 35.3 ms |
| Facet query | 8 | 2 / 0 | 14 | 5 / 2 | 33.3 ms |
| Facet on both sides | 9 | 2 / 1 | 11 | 3 / 3 | 33.2 ms |
| Batched facet on both sides | 9 | 2 / 1 | 11 | 3 / 3 | 27.0 ms |
| Dense six | 6 | 2 / 2 | 17 | 2 / 3 | 54.1 ms |

The batched route and dense two both issue 100 candidate pairs
per 50-query view. At the 5% ceiling, they each return two
grade-1 and zero grade-0 papers in the present view. The paired
strong-return difference is +10 points in favor of batched facets
(descriptive query-paper interval +1.9 to +20.8). At the 10%
ceiling the difference is −4 points (−12.2 to +3.9). The stricter
setting therefore supplies a specific operating point to recheck
on independent queries.

Batching reduces the facet route's present-view P95 from 33.2
to 27.0 ms, about 1 ms above the original dense-two route.
Shortlists and reranker score tuples agree across schedules in
47/50 present and 49/50 withheld views, with equal aggregate
grade counts at all tested ceilings. Do not assume bitwise
output equivalence. The experimental build uses 9,397 unique
compact vectors and takes about 34 seconds to construct both
the full and compact embedding indexes on this GPU. The
compact vectors are restricted to the known judged pools,
and the study does not establish storage or latency for a
production-scale corpus.

The evaluation measures paper relevance in known pools with
a supplied facet and 34 query papers. Attribution to individual
experts and generalization to enterprise requirements remain
separate tests. [Aggregate results](results.json)

## Reproduce

Use the pinned CSFCube files and the frozen Qwen checkpoints
from the [provenance manifest](../csfcube-lexical-challenger-v1/provenance.json).
Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.csfcube.facet_challenger \
  CSFCUBE_DIR benchmarks/csfcube-lexical-challenger-v1/provenance.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
