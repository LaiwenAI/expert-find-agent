# Experiment entry points

Current construction and maintenance code is grouped by purpose:

- `bilingual/`: grouped original needs, five-intent Chinese/English synthesis, full-original translation ablations, and frozen bilingual retrieval diagnostics.
- `metadata/`: publication intervals, paper citations and explicitly unknown individual contribution.
- `paper/`: separate English SIGIR main/supplement builds with artifact-backed or explicitly frozen result snapshots.
- `merit_factored_comparison.py` / `analyze_merit_factored.py`: complete paired native factorization ablation; a documented negative result, not the default method.

Frozen legacy experiment filenames remain at their original paths because stored protocol hashes, Slurm scripts and reproduction instructions refer to them. They are not separate production methods. Obsolete Markdown manuscripts and regenerable previews/bytecode have been removed. Further retirement must check protocol and import references, not just modification dates.

Do not delete raw inputs, active inference outputs, source provenance or failed scientific records as if they were caches. See `docs/usage.md` for commands and scope.

The exploratory `experiments.retrieval.cof_shortlist_ceiling` script computes relevance-label oracle ceilings for candidate pools on fully judged CoF SIGIR. For hard labels, 58 of 73 top-20 pools and nine top-50 pools have fewer than five positives; mean oracle P@5 is 0.608 and 0.962, respectively. The oracle uses test labels unavailable at inference and does not establish achievable accuracy or enterprise fit.
