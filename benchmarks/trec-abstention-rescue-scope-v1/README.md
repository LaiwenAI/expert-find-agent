# Separate rescue scope from verification budget after abstention

Use the two public TREC 2006 Enterprise qrels for experts and supporting
documents. Retain up to 20 judged experts per topic using a smoothed
expert relevance prior from the other 48 topics. Build a document
prior from person–document support judgments on those other topics.
The held-out topic's graded evidence is hidden until simulated
verification or evaluation. Its expert and evidence *pool
membership* is available in every arm. There is no live corpus or
unjudged document retrieval in this experiment.
[Official qrels](https://trec.nist.gov/data/t15_enterprise.html)

Shuffle score ties in 32 deterministic trials per topic. The initial
search verifies one source for each of the top five owners if a
judged source exists. When nothing is promoted, compare rescue
within five, ten or twenty ranked owners, under two or four
verification calls. Check each eligible pair at most once. Compare
the historical document prior, the same prior with distinct-owner
checks before additional sources for one owner, and a random
document order. All scope arms share initial calls and simulated
verifier outcomes. Evaluate exact support decisions and assumed
90% sensitivity and 5% false-positive rates, with errors either
independent by owner–document pair or correlated by document.

A grounded slot must have both an expert relevance grade of 2 and
an emitted citation with person-specific support grade 2. A
model-promoted slot that fails either condition counts as adverse
in the simulation, even if the citation itself supports that person.
Report candidate-specific checks and distinct source documents.
Resample whole topics after averaging over repeated orders. Mark
realized budgets that differ because pools run out of pairs.

## Results

Primary diagnostic uses the document prior with distinct-owner
checks and four rescue checks. All counts pool the 49 topics and
32 dependent trials. The five- and ten-owner scopes spent equal
checks in every trial for these conditions.

| Verifier | Within five grounded | Within ten grounded | Checks per scope, total | Change in supported P@5, 95% topic interval |
| --- | ---: | ---: | ---: | --- |
| Exact document support | 32 | 222 | 4,284 | +2.42 pp (+0.17 to +5.09) |
| Independent errors | 32 | 165 | 3,344 | +1.70 pp (+0.14 to +3.61) |
| Shared-document errors | 31 | 185 | 3,508 | +1.96 pp (+0.26 to +4.07) |

The denominator for these differences includes all 7,840 slots,
including queries that never needed rescue. With shared-document
errors, adverse promoted slots change from 192 to 162. These
are simulated ratings against known pooled references. Under
random document ordering, the shared-error gain is +0.68 pp
(+0.01 to +1.36), and under independent errors the interval
includes zero.

In the exact-support simulation with the document prior, 224 of
1,167 abstention replays have a grounded pair somewhere in the
ten-owner pool and none in the five-owner pool. This is an
opportunity calculation from all judged pairs, not a claim that
the four-check policy can find all of them. Extending to twenty
owners opens additional opportunities but also increases realized
checks in this replay. Keep the matched five-versus-ten result
as the scope contrast. Actual source recall, reader accuracy,
personal contribution, and latency require separate measurements.

[Aggregate results and diagnostics](results.json)

```sh
PYTHONPATH=src:. python -m experiments.trec_enterprise.abstention_rescue_scope \
  TREC_QRELS_DIRECTORY NEW_RESULTS.json
```
