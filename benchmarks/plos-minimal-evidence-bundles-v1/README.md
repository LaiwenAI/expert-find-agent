# Minimum citation bundles for expert search

Each query requires a conjunction of three publisher-declared CRediT roles. Return up to five identities in the fixed byline prior. For each person, find the smallest group of that person's sources that collectively supports all three roles. Break equal-size ties by ordered DOI. The [pinned PLOS fixture](../plos-credit-role-postings-v1/README.md) provides author ordinals, dataset-local contributor identifiers and licensed declarations from 152 articles.

For an unread author record, the roles declared across its article bound what it could establish. That bound cannot itself support a recommendation. Dynamic programming finds the provisional best bundle with unread records set to their upper bounds. If the best bundle is fully observed, it remains optimal with any later revelation. If there is no optimistic bundle, the expert cannot qualify in the snapshot. Otherwise the frontier method reads an unresolved source in the provisional optimum. The prefix control uses the same stopping rule and opens unresolved sources in DOI order. The eager control uses the same article bounds and opens all potentially relevant records before selecting a bundle.

Every route exactly reproduces brute-force expert order and minimum citations for all 364 role triples in both panels. On the complete 580-person panel, only 9 of the 1,820 returned expert rows require two sources. The 16-person linked panel contains 233 such bundles among 764 returned rows. Minimum bundles on that panel reduce distinct citations from 1,166 to 997 versus taking each role's first source. The linked panel intentionally enriches the cross-publication condition.

At zero author-cache coverage, the frontier opens 6,234 author declarations in the linked panel, versus 7,490 for prefix and 8,499 for eager. In the in-memory replay, this saves reads while increasing warm P95 from 29.2 to 33.4 microseconds relative to prefix. The timing covers the structured selection component and in-memory fallback only.

## Measured declaration loads

Materialize the pinned publisher declarations as 29 local per-article JSON files. Every file-based route shares decoded records across authors within one query. Repeat the linked-panel benchmark with actual file reads and JSON decoding. The resident control uses the eager algorithm with a memory lookup for the same declarations. File bytes are small (3.3 KB in total), and the filesystem is warm. All outputs and declaration-check counts reproduce the memory replay.

| Author-cache target | Prefix file opens | Frontier file opens | Eager P95 | Prefix P95 | Frontier P95 | Resident P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 6,473 | 5,357 | 122.8 us | 121.2 us | 115.9 us | 23.1 us |
| 50% | 3,076 | 2,848 | 63.6 us | 63.9 us | 66.5 us | 18.7 us |
| 75% | 1,326 | 1,098 | 36.4 us | 39.2 us | 39.5 us | 14.9 us |
| 100% | 0 | 0 | 10.7 us | 11.2 us | 11.2 us | 10.8 us |

At zero coverage the frontier reduces file opens by 17.2% and P95 by 4.3% relative to prefix. At 50% and 75% coverage fewer opens do not yield lower P95. Use resident declarations when readily available. The file benchmark measures the load of materialized declarations, including decoding. Publisher XML extraction, verification-model calls and storage service latency are separate stages.

## Reproduce

On a Slurm compute node:

```sh
PYTHONPATH=src:. python -m experiments.credit.minimal_evidence_bundles \
  benchmarks/plos-credit-role-postings-v1/sources.json NEW_RESULTS_JSON
PYTHONPATH=src:. python -m experiments.credit.minimal_bundle_materialization \
  benchmarks/plos-credit-role-postings-v1/sources.json SCRATCH_DIRECTORY NEW_FILE_RESULTS_JSON
```

These controlled conjunctions use known publisher roles, contributor IDs and a fixed prior. Results describe the scope of the visible snapshot. Free-form query routing and end-to-end search remain open. [Aggregate results](results.json)
