# Reuse qualification margins to guard packed rechecks

Full-context packing eliminates repeated copies of an
enterprise request in the independent BF16 fallback.
The [previous comparison](../enterprise-packed-clause-recheck-v1)
found qualification differences in two development requests.
Its guard recomputed every triggered block because it required
agreement with both cached precisions and a wide packed margin.

Keep the same BF16 and FP16 cached checks. For each clause
in a triggered block, express the packed, cached BF16, and
cached FP16 qualification margins in the direction of the
packed decision. The median is a two-of-three support margin.
Use the weakest clause's support as the block score. This
requires no separately trained model and reuses logits that
the search parser has already computed.

Calibrate by comparing packed and independent BF16 qualification
membership on 288 generated bilingual scope examples and 72
longer composites that join generated sections in one language.
The cutoff is the largest support score among calibration blocks
with any changed qualification, with a lower bound of zero.
If the calibration yields no such block, retain independent
BF16 for every fallback. The cutoff never uses the enterprise
development requests or the separately published heading controls.

Compare the original independent fallback, unguarded packing,
and the calibrated guard on the same 80 enterprise requests and
72 public synthetic heading controls. The enterprise requests
have no human qualification labels. Check every emitted
qualification, expert, and citation. Time complete warm
requests with rotating arm order and paired request bootstrap.
The measure includes retrieval and evidence verification.

## Results

Slurm job `43420` completed on one RTX 5090. Eight blocks
triggered an independent recheck in the 360 calibration
requests. Packing changed qualifications in three of those
blocks. The frozen cutoff was 0.5 and allowed one calibration
block to retain its packed output.

| Enterprise replay arm | Independent BF16 fallbacks | Fallback input slots | Full search P95 | Requests with qualification changes |
| --- | ---: | ---: | ---: | ---: |
| Original whole block | 7 | 74,680 | 994.3 ms | 0 |
| Packed block | 0 | 12,018 | 712.0 ms | 2 |
| Packed consensus | 6 | 80,106 | 1,005.0 ms | 0 |

The guard accepted one of seven triggered development blocks.
It reproduced every reference clause label, verification
decision, recommendation and citation in those 80 development
requests. The paired P95 change was +10.7 ms, with a descriptive
95% interval of −141.5 to +203.0 ms. All three routes issued
the same recommendations and citations on the same 37 requests.
The unguarded route changed intermediate qualification and
verification decisions in two requests. Its paired P95
difference interval crosses zero (−825.0 to +0.8 ms).

The guard still changed qualification membership outside the
anchored target in one of 72 independently published controls.
All three arms had the same 60 correct anchored labels.
Keep the original whole-block fallback. Reusing cached margins
removed one canonical forward but did not recover the latency
benefit of packing or preserve every qualification across both
evaluation sets. [Aggregate results](results.json)

## Reproduce

Use the authorized development data, pinned index and model
setup from the [previous packing study](../enterprise-packed-clause-recheck-v1).
Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_packed_consensus \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS ROLE_WINDOWS VECTORS.npy \
  VECTOR_METADATA MODELS benchmarks/enterprise-heading-roles-v1/controls.json NEW_RESULTS
```
