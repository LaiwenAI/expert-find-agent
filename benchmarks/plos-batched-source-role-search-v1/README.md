# Batch source discovery and contributor verification

Search the full 171-paper [PLOS source index](../plos-scoped-search-v1/README.md). Reuse the 120 explicit requests from the [source-scoped fixture](../plos-credit-source-scoped-v1/README.md). Two document clues can be result excerpts or keywords. The query supplies three contribution phrases and their intended source slots, but the search does not receive the source DOI. Evaluate each clue view with both sources present, and with either required paper excluded. Six source pairs drive this benchmark. Requirement variants repeat the same sources and clues.

Use the same dense and BM25 reciprocal-rank fusion and identity joins as the source-discovery reference. Compare linked top-four and rank-12 expansion with one side still in the first four. The shared Ridge mapping uses the frozen retrieval embedding, the existing 84 role training phrases, and the previously selected role-membership threshold. A person must be linked across both retrieved sources and their own roles must support each source-scoped requirement. The system sees publisher data and phrase positions. Target roles and source DOIs determine evaluation labels only.

| Encoder schedule | Forward calls |
| --- | --- |
| Separate | One source-clue batch and one task-phrase batch on every request |
| Joint batch | Combine the two source clues and three task phrases in a single forward with the original per-item prompts |
| Source first | Encode tasks only if source discovery finds a linked pair |

Use the separate path as the output equivalence control. Compare serialized source and owner results for every request. A recommendation counts as supported only if the exact expected source pair was found and the author appears in the gold contributor set. Wrong source substitutions count as unsupported even when the same author appears elsewhere. Report unique source-pair outcomes as well as decisions on repeated requirements. A known-role control is evaluated on the pair retrieved by each route.

Timing includes tokenization, online embedding, both retrieval scores, graph join, author role checks and serialization. The dense index, lexical index and role bank are resident. Network, model loading, raw XML parsing and offline construction are outside the timed path. The benchmark supplies source clues and task boundaries and measures a closed publisher index.

## RTX 5090 results

Job 42407 returns identical serialized source pairs and author sets for all 1,440 matched executions per schedule. The table gives the joint-batch path alongside the two-forward control.

| Both sources present | Source pairs found / 6 | Exact sources and authors / 120 | Requests with a supported return | Requests with an unsupported return | Separate → joint P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Result clues, linked top-four | 6 | 79 | 38 | 1 | 17.67 → 10.49 ms |
| Keyword clues, linked top-four | 2 | 25 | 12 | 1 | 16.99 → 8.84 ms |
| Keyword clues, linked expansion | 4 | 59 | 26 | 1 | 16.98 → 8.90 ms |

The split and joint schedules make 240 and 120 encoder calls, respectively, in each 120-request condition. Source-first uses 160 calls in keyword top-four and 240 in result top-four with both sources present, so enough queries still take the two-forward path to determine P95. For result top-four with either source masked, it uses 120 calls and lowers P95 to 9.09–9.16 ms, compared with 10.46–10.49 ms for joint batching. These measurements concern a single query with warm resident indexes.

For result clues, linked expansion adds no correct pair and reports 40 wrong pair selections over 240 missing-source request versions, leading to 11 unsupported author returns. Linked top-four reports zero wrong pairs or unsupported people in those controls. For keyword clues, both policies abstain on both missing-source conditions throughout this small cohort. The earlier separate 4B role parser [results](../plos-scoped-search-v1/results.json) score 90/120 exact source-and-author outputs with result clues and 71/120 with keyword expansion, compared with 79 and 59 for the shared ridge route. Its full-route timings are unavailable on the same protocol. Encoding consolidation preserves output within the same ridge route, while the model comparison retains a measured quality gap. [Aggregate results](results.json)

## Reproduce

Use the pinned [publisher projection](../plos-scoped-search-v1/README.md), NISO phrases, synthetic training descriptions, verified Qwen3-Embedding checkpoint and [shared witness gate results](../plos-shared-embedding-witness-v1/results.json). Run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.batched_source_role_search \
    SCOPED_FIXTURE NISO_EXAMPLES PUBLISHER_PROJECTION ROLE_TRAINING QWEN_EMBEDDING PRIOR_RESULTS NEW_RESULTS
```
