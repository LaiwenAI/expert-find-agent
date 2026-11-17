# Bilingual query construction dataset

The local dataset is `artifacts/dataset-release/bilingual-query-construction-v1/`. It contains 1,000 complete original Chinese enterprise needs and 5,000 derived query intents with parallel Chinese/English text. Full English translations of the original needs remain pending. Parallel language views do not increase the independent-query count.

The five intent types are overview, component method, conditional transfer, evidence lookup, and constraint-focused search, with one generated slot per type and parent. Parents come only from the construction split of the new company/body/URL-grouped cohort. Validation and sealed groups were not used. Domain categories and source record hashes are retained for original queries; intent types describe generated requests rather than independently adjudicated labels.

Generation and native grounding/equivalence review use `gpt-6-luna`. All 1,000 parent jobs reached terminal records and all 5,000 slots contain generated text. The original screen retained 4,344 pairs. A separate conservative numeric postcheck retains 4,224 and downgrades 120; the final export retains all 776 unjudged pairs. There are **zero expert-fit labels**. Screened status does not establish human-verified semantic equivalence, relevance, or actual expert capability.

Version-2 numeric checking handles additional negated comparator aliases and abstains on unsupported nearby negation. Across all pairs it reports 4,400 numeric matches, 370 mismatches and 230 uncertain checks. A numeric match is insufficient for admission: the export intersects it with the previous grounding screen and never promotes a prior failure. Numeric-free pairs have no numeric evidence of equivalence. Source responses, the frozen original generator and its version-1 guard remain unchanged.

Rebuild locally after generation has completed:

```bash
PYTHONPATH=src python -m experiments.bilingual.postvalidate_queries
PYTHONPATH=src python -m experiments.bilingual.export_dataset
```

The exporter rejects changed source cohorts or parent records, incomplete parent coverage, missing postchecks, duplicate query slots, and protected-split entries. Each output has a SHA-256 manifest. Original enterprise text remains subject to its original access terms and is excluded from the public MIT software archive. This is a construction dataset, not a completed independently labeled expert-search benchmark.


A subsequent full-original translation pilot compares direct, copied-placeholder, and fixed-numeric-block gap translation on the same 32 construction parents. Combined native/limited-numeric screening retains 8, 6, and 3 records respectively; unresolved request/structure counts are 1, 9, and 8. The runs were sequential exploratory ablations, and screens are not human gold. No variant is adopted for translating all 1,000 original needs. The 5,000 derived bilingual requests remain a separate construction product.

A frozen retrieval diagnostic selects one derived query per original parent, cycling types to give 200 each: 860 retain the screen, 140 remain unjudged. Exact Chinese/English top-ten overlap is 42.16%; this is ranking stability, not relevance accuracy. HNSW floor512 retains99.805% of exact top-ten entries across2,000views but only1,962 ordered lists match. These results motivate further alignment and verification, not a claim of bilingual quality completion.

The original 2,741-need pool has 1,395 group IDs, with no group crossing construction, validation and sealed splits. Whitespace-stripped character four-gram Jaccard screening of 513 validation/sealed needs against construction gives maximum 0.329; no maximum reaches 0.5. This aggregate read-only diagnostic does not detect semantic paraphrases, shared organizations missed by grouping, or other leakage. The reproducible check is `experiments/bilingual/check_split_overlap.py`; protected examples remain unavailable for tuning.
