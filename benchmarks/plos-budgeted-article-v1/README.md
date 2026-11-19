# Evidence-driven article reads under a fixed budget

Use 364 three-role queries, the published declarations of 580 PLOS contributor identities across 152 articles, and the same fixed byline prior and minimum-bundle rule as the [unbudgeted experiment](../plos-minimal-evidence-bundles-v1/README.md). A linked panel keeps the 16 identities with multiple articles to stress cross-source evidence. Only nine of the 1,820 reference positions in the full panel require multiple citations, versus 233 of 764 in the linked panel.

A source read opens one warm local JSON file and exposes the declarations of all its authors, once per query. Every route starts with the same article role unions and either zero or roughly half the declarations cached. A union bounds the roles a source can establish and cannot support an individual by itself. The observed declarations build the actual citation bundles. Read budgets are two, four, and eight article files. The oracle reference ranks people with all declarations visible.

- **Fixed prior:** Order initially useful articles by the best candidate they can help and break ties by article ID.
- **Fixed shared coverage:** Also count the possible gains for several candidates, discounted by optimistic rank and the size of the smallest cover.
- **Adaptive frontier:** Recompute which unseen sources could complete each candidate within the remaining budget. Read from a smallest cover for the highest-ranked unresolved candidate that may still affect Top-5.
- **Adaptive shared coverage:** Recompute the shared coverage score after each read.
- **Incremental frontier:** Keep per-candidate plans in a priority queue. Refresh plans only for co-owners of the newly read article. Reject invalidated plans and those that no longer fit the remaining budget.

## Results

The table uses the linked panel with zero cached declarations. Recovery counts positions from the exhaustive reference Top-5. Latency includes local file loading, JSON decoding, scheduling and citation selection. Timings come from one CPU allocation on ws5 and exclude fetching publisher sources, building the index, parsing a natural-language query and model-based verification.

| Budget | Route | Recovered reference positions / 764 | Exact order and citations / 364 | P95 |
| --- | --- | ---: | ---: | ---: |
| 4 | Fixed prior | 246 | 132 | 43.2 us |
| 4 | Fixed shared coverage | 242 | 127 | 43.5 us |
| 4 | Adaptive frontier | 308 | 152 | 79.2 us |
| 4 | Adaptive shared coverage | 306 | 152 | 104.9 us |
| 4 | Incremental frontier | 308 | 152 | 50.5 us |
| 8 | Fixed prior | 510 | 193 | 61.9 us |
| 8 | Incremental frontier | 604 | 246 | 76.2 us |

The incremental frontier matched every read and ranked citation of the adaptive frontier across all 4,368 query/settings combinations. At four reads it recovered 62 more reference positions than the fixed prior in the linked panel and reduced P95 by 36.2% relative to recomputing the frontier. The fixed prior was faster. The full panel showed no recovery gain from either frontier and usually favored the fixed prior for latency. At 50% cache coverage in the linked panel, the four-read incremental frontier recovered 623 positions versus 618 for the fixed prior. Shared coverage added cost and produced mixed recovery changes. The 364 requests enumerate role triples on the same snapshot, so they are not independent held-out queries. All returned recommendations were checked for complete support by the cited person's publisher-declared roles.

## Reproduce

Job 41136 runs the [benchmark](../../experiments/credit/budgeted_article_evidence.py) on a Slurm compute node. Use the same source fixture and a scratch directory on the compute node. [Aggregate results](results.json) include all budgets, cache settings and both panels.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.budgeted_article_evidence \
  benchmarks/plos-credit-role-postings-v1/sources.json SCRATCH_DIR NEW_RESULTS
```
