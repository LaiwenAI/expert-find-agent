# Match enterprise rescue against hybrid retrieval and late verification

Use the existing 80 development queries and 52,667 owned sources. No
sealed demand is opened. Fit the lexical index on the source corpus
and reuse frozen source vectors. Use the same qualification parser,
two source excerpts per candidate, reader, source visibility and
candidate cutoff across methods. All arms use open-weight models.

| Route | First candidate pool | On first-batch abstention |
| --- | --- | --- |
| Dense | Exact maximum source similarity | Stop |
| Hybrid | BM25 and exact dense, fused by owner | Stop |
| In-pool role rescue | Same hybrid pool | Verify up to two declared-role passages within the original pool |
| Global role rescue | Same hybrid pool | Verify up to two declared-role passages from the full source index |

The hybrid route fuses the first 50 owners from each lexical and dense
channel with reciprocal rank fusion (constant 60), then fuses each
retained owner's source list within the same two-source cap. A
channel-only expert can supply fewer than two sources. Report the
actual number of verification pairs as well as the common caps. The
rescue reads share the same passage index, retrieval score, acceptance
rule, and cap. Because one scope can run out of eligible passages, the
equal nonzero pair subset is a retrospective paired diagnostic.
Record exact evidence views, not just the recommended experts.

The timed interval includes qualification parsing, encoding,
retrieval, verification, and fallback. Record offline lexical build
cost separately. This is warm serial execution on one worker.
Queueing, network transport, and sustained capacity require a
separate experiment.

Recommendation counts describe model-issued results. The exact-view
review uses the original demand, the candidate and the exact citation
spans, with method and rank withheld. Independent reviewers judge
method fit, all applicable prior qualifications, and personal
contribution. Cached proprietary text and private traces stay on the
authorized systems. Public results contain only aggregate counts,
settings, and provenance fingerprints.

## Results

Job `43546` completed on one RTX 5090. The lexical index took 19.04
seconds to build and its resident posting arrays occupy 61.0 MB.
The following counts reflect model outputs before human review.

| Method | Issued requests / 80 | Issued slots | First-pass checks | Extra checks | Warm serial P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 34 | 70 | 1,280 | 0 | 1,117 ms |
| Hybrid | 25 | 51 | 1,279 | 0 | 1,153 ms |
| Hybrid + in-pool role rescue | 27 | 53 | 1,279 | 128 | 1,152 ms |
| Hybrid + global role rescue | 30 | 56 | 1,279 | 206 | 1,191 ms |

The paired descriptive P95 change for global rescue against hybrid
is +14.77 to +99.11 ms (95% request bootstrap interval). Global
rescue adds five distinct slots, four outside the initial shortlist.
Among 36 requests where both rescue routes spent the same nonzero
verification budget, the two routes each spent 120 checks in total.
Global rescue issued a result for four requests, in-pool rescue for two.
This subset is selected by realized cost, and issued counts do not
establish the relevance or personal ownership of any source.

The privately frozen packet has 109 exact views. The global route
shares all 51 hybrid views and adds five. Against in-pool rescue
it shares 52 views, adds four and loses one. On a fixed denominator of
400 Top-5 slots, its strict supported P@5 gain lies between 0 and 1.25
percentage points against hybrid and between −0.25 and +1.00 point
against in-pool rescue while all labels are missing. These are exact
finite-panel bounds from unresolved views. The realized gains and
sampling uncertainty remain unknown. The original dense baseline should
also remain in the comparison because hybrids changed most evidence
and yielded fewer provisional outputs on this cohort.

[Aggregate timing and counts](results.json) ·
[Exact-view audit](blind-audit.json)

## Reproduce

Run on a Slurm CUDA allocation with a private scratch directory:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_hybrid_rescue run \
  DEVELOPMENT_QUERY_PROJECTION DEVELOPMENT_CORPUS_PROJECTION ROLE_WINDOWS \
  VECTORS VECTORS_MANIFEST MODEL_ROOT PRIVATE_SCRATCH NEW_AGGREGATE NEW_PRIVATE_TRACE
```

Generate and validate the review locally, using the verified development
projection and original frozen dataset manifest:

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_hybrid_rescue packet \
  DATASET PRIVATE_TRACE AGGREGATE QUERY_PROJECTION CORPUS_PROJECTION NEW_PRIVATE_PACKET
PYTHONPATH=src:. python -m experiments.retrieval.audit_enterprise_hybrid_packet \
  PRIVATE_PACKET QUERY_PROJECTION AGGREGATE NEW_PUBLIC_AUDIT
```
