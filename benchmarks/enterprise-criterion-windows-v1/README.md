# Verify enterprise criteria with contiguous source windows

Transfer the public [claim-specific evidence comparison](../redocred-claim-bound-windows-v1)
to the live enterprise development route. Reuse the pinned 80
requests, 52,667 sources, parser, exact dense index, original top
five experts, and two source records per expert. The same guarded
packed reranker processes the same number and order of primary
criterion–source pairs across all arms.

| Arm | Source text shown for each criterion |
| --- | --- |
| Reference | The first 800 characters, as in the earlier guarded route |
| Short lead | The first 320 characters of the same source |
| Criterion window | The 320-character contiguous span within the original opening with most matching criterion terms |

Criterion-window candidates start at the beginning, sentence
boundaries, a 160-character grid, and the final aligned window.
Chinese character bigrams and Latin tokens share the existing
tokenizer. Break ties at the earliest offset. All selected spans
refer to original source offsets. Candidate and source
nominations are fixed before shortening. The abstention-triggered
role-index fallback is unchanged. It can receive a different
set of issued experts or be called on a different request.

Run each arm live on every request and rotate the order. Measure
primary verification and the full serial route. Compare issued
expert identities and ordering, primary source IDs, verified
spans, and fallback outputs. A source ID match is weaker than
a span match. Replacing a span can remove the qualifying context
even when the source and expert remain the same.

These counts describe agreement with the existing model pipeline.
No candidate or citation disagreement is labeled correct or
incorrect without independent review. The sealed requests are
unused, and the release contains aggregate data only.

## Results

Slurm job 43368 completed on an RTX 5090 on ws5. Each
arm checks 1,280 primary pairs and 184 fallback pairs. All
three routes issue a recommendation on the same 37 requests,
and fallback invocation and output agree on all 80.

| Arm | Primary executed tokens | Primary P95 | Full serial P95 | Requests with changed expert lists | Requests with changed primary sources / spans |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 385,612 | 114.5 ms | 1,069.0 ms | 0 | 0 / 0 |
| Short lead | 291,104 | 95.2 ms | 1,052.0 ms | 7 | 13 / 18 |
| Criterion window | 291,434 | 98.1 ms | 1,057.0 ms | 7 | 14 / 18 |

The criterion route lowers primary executed tokens by 24.4%
and primary P95 by 14.3%. Its full serial P95 is about 12 ms
lower in this replay. Each short route gains four individual
expert slots and loses four relative to the reference. Equal
issued-query counts conceal changes to specific recommendations.
The criterion selector does not clearly improve on the short
lead under the same span budget. Direction and uncertainty
of quality differences remain unresolved until blinded
review, so retain full openings for issued claims while
testing shorter windows as proposals. [Aggregate results](results.json)

## Reproduce

Use the authorized [enterprise development resources](../enterprise-role-first-query-v1)
and pinned models on a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_criterion_windows \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS ROLE_WINDOWS VECTORS.npy \
  VECTOR_METADATA MODELS NEW_AGGREGATE_RESULTS
```
