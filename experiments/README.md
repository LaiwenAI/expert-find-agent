# Experiment entry points

Current construction and maintenance code is grouped by purpose:

- `bilingual/`: grouped original needs, five-intent Chinese/English synthesis, full-original translation ablations, and frozen bilingual retrieval diagnostics.
- `metadata/`: publication intervals, paper citations and explicitly unknown individual contribution.
- `paper/`: separate English SIGIR main/supplement builds with artifact-backed or explicitly frozen result snapshots.
- `merit_factored_comparison.py` / `analyze_merit_factored.py`: complete paired native factorization ablation; a documented negative result, not the default method.

Frozen legacy experiment filenames remain at their original paths because stored protocol hashes, Slurm scripts and reproduction instructions refer to them. They are not separate production methods. Obsolete Markdown manuscripts and regenerable previews/bytecode have been removed. Further retirement must check protocol and import references, not just modification dates.

Do not delete raw inputs, active inference outputs, source provenance or failed scientific records as if they were caches. See `docs/usage.md` for commands and scope.

The exploratory `experiments.retrieval.cof_shortlist_ceiling` script computes relevance-label oracle ceilings for candidate pools on fully judged CoF SIGIR. For hard labels, 58 of 73 top-20 pools and nine top-50 pools have fewer than five positives; mean oracle P@5 is 0.608 and 0.962, respectively. The oracle uses test labels unavailable at inference and does not establish achievable accuracy or enterprise fit.

`experiments.retrieval.cof_field_union --output <new-directory>` explores a fixed 20-reviewer budget combining title and abstract candidate-LM rankings, filling remaining slots from the complete query, then reranking by BM25. On the reused CoF SIGIR test set, hard P@5 is 0.170 versus 0.156 for full-query top-20, with a paired query-bootstrap difference interval of [-0.005, 0.033]. Hard pool recall is 0.100 for both to three decimals. One timing per query excludes index construction, initialization, evidence selection and reasoning. These post-hoc results do not establish improvement, and reviewer matching is not enterprise fit.

`experiments.retrieval.cof_field_union_kdd --output <new-directory>` tests a 10-candidate version on 174 KDD queries, restricting retrieval to each query's 20 already-judged reviewers. In the 148 queries with a hard positive, mean pool recall is 0.568 for full-query top-10 and 0.569 for the field union; the paired difference interval is [-0.030, 0.033]. Single-query mean times are 1.67 and 3.35 ms, excluding model construction and initialization. This judged-candidate restriction is benchmark supervision, not a deployable global retrieval test. Label-oracle ceilings are not achievable ranking accuracy, and near-duplicate leakage remains possible.

`experiments.analyze_evidencebench_set_efficiency --output <new-directory>` rechecks saved EvidenceBench development outputs, their frozen dataset and native-candidate hashes, and coverage calculations. Across 37 attempts, set selection and independent Decisions obtain macro aspect recall 0.590 and 0.562, and micro recall 0.523 and 0.502. The two failed set attempts retain the semantic fallback. Recorded mean input tokens are 7,190 and 14,114, respectively, but fields and API interfaces differ; historical sequential timings do not establish a speedup. This is not a SetR checkpoint reproduction, expert-fit assessment, independent test, or semantic-support validation.
