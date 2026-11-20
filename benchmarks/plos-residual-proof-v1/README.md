# Verbatim residual proof under early stopping

This replay connects the request-packet builder to the fixed-pool
constraint verifier. It reuses 108 held-out views of 18 contributor
edits across three publisher-linked components from the
[PLOS ONE compositional fixture](../plos-credit-compositional-edits-v1/README.md).
It renders three hard conditions and one explicit preference per request.
Candidate identities, order, allowed witnesses, and judgments come from
the fixture. Canonical source order is declared by the benchmark.
The replay supplies these interpretations and judgments directly.

Run each case with the first clause marked as a trusted role
interpretation, then without any role interpretation. The marked
clause still requires evidence from the candidate. Both layouts
produce the same ordered experts, original requirement offsets,
canonical citations, and work counts. A separate 36-view intervention
per scope and cutoff masks the final criterion judgment in the second
publication. A missing judgment has no evidentiary force.

## Results

Counts below cover the 108 original views per publication scope and
cutoff. Each source read reveals the available criterion judgments for
one author–publication pair. Every proof-preserving arm returns the
same experts in order with the same clause–source citations as exhaustive
reading. Adaptive refutation saves zero reads over width-aware fixed
ordering, including in the masked-declaration intervention.

| Scope | Cutoff | Exhaustive reads | Request-order reads | Width-aware fixed reads | Adaptive reads | Exact full candidate ledgers under adaptive stopping |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Specified publication | 1 | 264 | 80 | 64 | 64 | 0 / 108 |
| Specified publication | 3 | 264 | 96 | 78 | 78 | 6 / 108 |
| Either publication | 1 | 264 | 199 | 199 | 199 | 63 / 108 |
| Either publication | 3 | 264 | 264 | 264 | 264 | 108 / 108 |

With both specified publications visible, Top-1 uses 64 reads versus
132 exhaustive and 80 in request order. The 72 withheld-publication
views in this scope all abstain without reads in the stopping arms,
because no candidate has a complete witness inventory. They are
included in the table. Retaining only the interpreted role yields 72
recommendations in the specified-publication Top-1 replay, of which
52 lack full proof. Even with both publications visible, 16 of 36
recommendations from that ablation lack full proof.

For the masked-declaration intervention with either publication allowed,
Top-1 reads 116 source bundles versus 132 exhaustive and returns the
same 12 experts with citations over 36 requests. On 10 of those
requests the full ledgers differ. Fourteen candidate conditions that
exhaustive reading labels unavailable remain unassessed under early
stopping. The ledger also tracks conditions that exhaustive reading
would support or reject. A runtime trace may report a condition as
unassessed when relevant sources remain unread. A complete diagnostic
ledger requires further reads when an unread result can change its
status, even when it cannot change the recommendation.

These are work counts for an in-memory replay with known source scope
and fixed judgments. The fixture tests held-out publisher declarations
with templated clauses. Free-text interpretation, dynamic candidate
generation, retrieval latency, and expert-level relevance need separate
evaluation. Earlier [live reader controls](../constraint-certificate-local-v1/README.md#live-open-model-control)
show why fewer nominal reads require a separate latency comparison
against strong fixed schedules and batching.

[Machine-readable results](results.json)

```sh
PYTHONPATH=src:. python -m experiments.credit.verify_residual_requests \
  benchmarks/plos-credit-compositional-edits-v1 /path/to/new-results.json
```
