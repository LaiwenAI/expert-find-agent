# Repeat same-person expert nomination with independent MEDLINE indexing

Keep the author-disjoint [322-article eLife catalog](../elife-role-nomination-v1).
Resolve public article DOIs using [NCBI ESearch and EFetch](https://www.ncbi.nlm.nih.gov/books/NBK25499/).
Require exact case-insensitive DOI matches in the retrieved records and
retain only IDs, indexing status and descriptors in the released snapshot.
A missing DOI or absent descriptor is not proof of irrelevance.

Use [Medical Subject Headings](https://www.nlm.nih.gov/mesh/meshhome.html)
as a metadata signal separate from author keywords. First, audit the
previously frozen eLife keyword citations. Restrict this check to
keyword strings that exactly match a descriptor attached to at least
one MEDLINE record in the corpus. Report exact and exact-major matches
as two positive signals. No synonym expansion or negative labels are
inferred.

Second, use the major-descriptor names as queries and the associated
MEDLINE article assignments as topic-positive labels. Keep descriptors
that occur on 4–40 articles across the full 322-source catalog, excluding
six routine population descriptors by ID. For each topic choose the two
partially supported CRediT role pairs by the original fixed procedure.
The entire catalog is still searchable, including documents without
MEDLINE indexing. Neither MeSH labels nor author keywords are article
features in retrieval and verification. As in the original eLife
experiment, a recommendation must bind its two roles to the same author
in the cited article.

Compare the standard four- and eight-source topic controls, unrestricted
and top-eight-constrained person nomination, and the
[PLOS-calibrated topical floor](../role-score-gate-v1). All policies reuse
the existing open checkpoints and verifier cutoff. Import the frozen
margin unchanged and report both role-declaration masks. This evaluates
a new metadata-derived task over the same public documents. It does not
add human excerpt judgments or validate enterprise suitability.

## Results

All 322 DOIs mapped to distinct PubMed IDs with MEDLINE status in
the snapshot taken on 2026-11-20. Of the 24 original keyword topics,
15 matched an indexed descriptor exactly, covering 30 of 48
requests. At full available declaration coverage, person-four
nomination raised issued slots from 65 to 81 in that subset.
Exact descriptor-supported slots rose from 35 to 40. Exact major
descriptor-supported slots stayed at three. These annotations apply
to the article as a whole and may omit legitimate excerpt support.

RTX 5090 jobs `43795` and `43797` reproduce all non-timing metrics.
The original role-pair construction yields ten requests across five
eligible major descriptors. All policies issue recommendations for
nine of the ten. In the full available mask, the strongest tested
topic control and the nomination variants yield:

| Policy | Tagged / off-label slots | Reads | Local P95, job 43795 |
| --- | ---: | ---: | ---: |
| Topic four | 18 / 8 | 40 | 25.13 ms |
| Topic eight | 19 / 13 | 80 | 37.79 ms |
| Person four | 19 / 11 | 40 | 25.15 ms |
| Top-eight person four | 19 / 11 | 38 | 25.17 ms |
| PLOS-calibrated topical floor | 19 / 10 | 40 | 25.19 ms |

At the sparse mask, the four-source methods all return 18 tagged
slots. The gain of one marked recommendation under the full mask
has a descriptive topic-cluster 95% interval from 0 to +6 percentage
points of P@5 versus topic four. The narrow sample gives weak evidence
about transfer to new topics and cannot establish cross-domain
recommendation accuracy. Off-label citations are unresolved. Timing
is local warm execution and excludes service queues.

[Keyword audit](keyword-audit.json) · [Major-descriptor replay](results.json)

## Reproduce

Rebuild the pinned eLife projection as described in the
[original transfer experiment](../elife-role-nomination-v1). Use the
released snapshot for replay. A fresh fetch from the live API may
return updated records. `capture` is provided for collection provenance.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.elife_mesh capture ELIFE_PROJECTION NEW_SNAPSHOT
python -m experiments.credit.elife_mesh audit benchmarks/elife-mesh-v1/mesh.json \
  ELIFE_ASSIGNER_PACKET benchmarks/elife-role-review-v1/results.json NEW_KEYWORD_AUDIT
python -m experiments.credit.elife_mesh run ELIFE_PROJECTION benchmarks/elife-mesh-v1/mesh.json \
  benchmarks/elife-role-nomination-v1/results.json benchmarks/role-score-gate-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
