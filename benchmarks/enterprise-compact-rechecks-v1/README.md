# Compact rechecks of cached enterprise clause decisions

Use the frozen 2.0-logit recheck threshold from the [whole-block benchmark](../enterprise-guarded-parsing-v1/README.md) on the same 72 public controls and 80 private development requests. The model, full-context prompt, source anchors and initial cached passes are shared across all three recheck arms. The standalone full batch is the role-agreement reference.

- Block recheck evaluates every clause in any flagged block.
- Row recheck evaluates only flagged clauses within their original blocks.
- Coalesced recheck sorts the flagged clauses by length and batches them within each request.

Each route regenerates its own predictions during timing. Rotating arm order balances sequential runs. Measurements include tokenization, cache setup, all model calls and result extraction. They exclude retrieval and downstream verification. Packing fewer rows changes GPU batch geometry, so agreement with the reference is measured directly.

## RTX 5090 results

| Mode on 80 development requests | Rechecked clauses | Extra batches | Clause disagreements / 1,452 | Qualification additions / removals | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent full batch | 0 | 0 | 0 | 0 / 0 | 5,269 ms |
| Prefix cache | 0 | 0 | 16 | 2 / 1 | 443 ms |
| Whole-block recheck | 422 | 56 | 0 | 0 / 0 | 3,282 ms |
| Row recheck in original blocks | 71 | 56 | 18 | 5 / 1 | 830 ms |
| Coalesced row recheck | 71 | 34 | 15 | 5 / 2 | 887 ms |

Rechecking just 71 clauses lowered the tail latency of whole-block rechecking, but qualification disagreements increased under both compact schedules relative to the cache-only arm. Whole-block rechecking was the only tested recheck that reproduced all independent labels. Each mode correctly classified 60 of 72 public target clauses. Those controls also supplied the fixed margin in the previous study. The separate development packet contains 26 exact disagreements across 15 requests; it has no human judgments. [Aggregate results](results.json)

## Reproduce

Inside a Slurm CUDA allocation, use the same development projection and model files as the linked benchmark:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_compact_rechecks \
  DEVELOPMENT_JSONL benchmarks/enterprise-heading-roles-v1/controls.json \
  QWEN_CHECKPOINT NEW_AGGREGATE --disagreements NEW_PRIVATE_TRACE
PYTHONPATH=src:. python -m experiments.retrieval.export_compact_recheck_review \
  ORIGINAL_QUERY_JSONL NEW_AGGREGATE NEW_PRIVATE_TRACE NEW_PRIVATE_PACKET
```

The review exporter verifies source and trace hashes, then shows each exact disputed clause in its request without model labels or method identifiers. The private trace, method map and review form stay local. Independent judgments are required to assess semantic correctness.
