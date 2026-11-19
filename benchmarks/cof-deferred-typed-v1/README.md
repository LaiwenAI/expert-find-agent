# Defer 4B evidence checks until they can change the ranked output

Reuse the frozen CoF panels, candidate generator, and generic and typed verifiers from the [live confirmation study](../cof-typed-evidence-v1/README.md). All three arms independently encode the query and retrieve the same ten candidate reviewers and up to three sources per candidate. Each confirmed reviewer requires two accepted citations. The comparison runs all stages, including retrieval and both models, with warmed weights and rotating arm order.

The immediate policy checks every generic positive in the fixed margin band when it is first seen. The deferred policy applies the same final per-source rule. It marks a positive in that band tentative, then completes model-cheap checks in the optimistic response. When only tentative citations can still separate the pessimistic and optimistic outputs, it calls the 4B verifier on those sources. The search terminates when the ranked reviewer IDs and citation lists match under both bounds. A generic-negative or high-margin generic-positive verdict stays definitive as before. The generic-only arm sets a lower-latency reference.

With fixed, context-independent decisions, scheduling changes which sources are inspected and when, while the stopping rule certifies the same ranked response. A synthetic check compared both implementations with exhaustive labels on 1,000 shared-source inputs. Live batching can change numerical values, so measure exact equivalence of returned experts and citations separately. Published CoF hard labels cover some reviewers and provide no independent citation judgments.

## RTX 5090 results

| Panel | Schedule | Generic forwards | 4B forwards | 4B sources scored | Full P95 | Exact output vs. immediate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KDD | Generic only | 5.89 | 0 | 0 | 261.49 ms | 36 / 122 |
| KDD | Immediate | 5.85 | 2.66 | 4.20 | 612.20 ms | 122 / 122 |
| KDD | Deferred | 7.21 | 1.10 | 2.96 | 567.43 ms | 117 / 122 |
| SciRepEval | Generic only | 6.90 | 0 | 0 | 263.96 ms | 21 / 59 |
| SciRepEval | Immediate | 6.95 | 2.85 | 4.68 | 590.70 ms | 59 / 59 |
| SciRepEval | Deferred | 8.22 | 1.19 | 3.25 | 577.57 ms | 56 / 59 |

Job 42112 reproduced the generic and immediate arms' prior aggregate counts. Deferral reduced mean 4B forwards by about 59% and scored 4B sources by about 30%, while increasing generic forwards. Query-paired mean time savings had 95% intervals of 14.66 to 31.31 ms on KDD and 17.55 to 38.42 ms on SciRepEval. P95 was still more than twice the generic-only arm on either panel.

Live ranked responses and citations differed from immediate confirmation on eight queries. Changes in batch shapes can affect either model stage, and the aggregate comparison does not resolve the source of the divergences. Cited known positives were unchanged at eight on KDD and fell from 18 to 17 on SciRepEval. Cited judged negatives increased from 21 to 22 on KDD and stayed at eight on SciRepEval. The software certificate applies to fixed verdicts. The live experiment demonstrates reduced verification work with residual numerical or execution discrepancies that require diagnosis before claiming output-preserving acceleration. [KDD aggregates](kdd.json), [SciRepEval aggregates](scirepeval.json)

Run the [confirmation study commands](../cof-typed-evidence-v1/README.md) with `--compare-deferred` and fresh result paths inside a Slurm CUDA allocation.
