# Requirement-Anchored, Same-Author Source-Control Data

Current local version: `artifacts/dataset-release/paper-controls-combined-v3`, 5 pairs, 10 views. Each pair contains a full-source view and a same-author paper-replacement view; labels are `supported` and `unknown`, and describe only whether the snapshot supports the specified narrow condition. `unknown` does not mean expert incompetence and is not a relevance negative. All labels are silver labels from two models from the same provider; there are no human gold labels.

| Construction cohort | Pair count | Selection method |
|---|---:|---|
| development | 2 | Early exploration of domain overlap and paper sources |
| new_cohort | 2 | Fixed sampling after excluding early demand/expert IDs; still limited by domain overlap and paper availability |
| demand_first | 1 | Reuse the original fixed 20 demands; take the top-ranked expert by paper-vector search across the full corpus and that expert's top three sources |

The demand-first funnel is retained in full: 20 demands -> 20 retrieval tasks -> 9 construction candidates -> 4 candidates supported by both models -> 3 pairs for which a genuine replacement view can be constructed -> 1 pairs accepted by both models for the control. The 11 tasks that cannot be directly constructed do not mean there are no potentially suitable experts; the 5 span-review rejections, 1 cases missing a replacement source, and 2 control-review disagreements were not deleted or changed to definitive labels. Construction yield is not retrieval accuracy.

Records retain the complete original demand, narrow condition, remaining demand gap, source attribution and snapshot, citation offsets, model reviews, cohort, and parent-artifact hashes. examples SHA256: `a2bd29e467c3a4945bcbd96fe7554f05a102bfcdb761f3f984b71dd7bc1bf52a`. Across the three cohorts, exact overlap in expert IDs, demand IDs, normalized conditions, and full source text is zero for every pair of cohorts; semantic near-duplicates have not yet been checked. Do not declare the data leakage-free based only on exact deduplication.

There is currently no train/test split. Paired views are not independent samples; future splitting must use connected groups formed by demand, expert, and shared sources, and retain all construction-time rejections and disagreements. The sample is too small for conclusions about statistical significance, generalization ability, or system SOTA. Evidence for facts in the literature and applicability to future demands must be evaluated separately; this binary label must not be used as a gold label for overall project capability.

This data is currently only a local research artifact and contains original enterprise demands and excerpts from instructor papers. The MIT software license does not cover these sources; the data has not been publicly released, and documentation of authorization to redistribute the data sources is incomplete. The source-review package contains code and this card, but not the original text, data samples, or model traces. The public benchmark adapter and this synthetic data are separate deliverables.
