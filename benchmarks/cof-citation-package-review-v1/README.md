# Blind review of citation packages from full-inventory CoF search

Replay the equal-blend and citation-only interventions from [the shared-paper experiment](../cof-ownership-specificity-v1/README.md) over frozen KDD and SciRepEval. The same five expert identities and ordering are an invariant. The exporter checks that both arms reproduce the published counts of cited candidates and cited known positives before sampling.

For each dataset, sample up to 16 candidate slots where the two returned citation packages differ and both are nonempty, eight where only one arm returns evidence, and eight unchanged packages as controls. Sampling uses a fixed hash without consulting relevance grades. For a candidate and query, the same citation appearing under both arms receives the same case identifier. The displayed order hides arm, rank and known relevance grade. Each row shows the query and the exact cited title and abstract. It also shows up to three independent publication titles from the indexed reviewer profile. Titles with lower reviewer overlap come first. These are identity lookup anchors, and cited papers are excluded from the anchors.

The form requires separate judgments about content support and the candidate's personal contribution. When contribution is determinate, reviewers record the original publisher's statement and public evidence that establishes candidate identity. Affiliations, surnames and the dataset's membership edge alone are insufficient. The script checks fields, quotation placement for indexed content and source URL structure. The reviewer and adjudicator assess the meaning of external source material. Cases without uniquely recoverable identity remain insufficient for contribution.

Two separately attributed copies are required. The scorer keeps missing or disputed labels unresolved, along with agreed insufficiency. A package counts as clean when it is nonempty and every citation supports the query and the selected expert's contribution. The exact paired bounds assign shared unresolved labels identically across both arms. These are identification bounds over sampled slots, not sampling confidence intervals or estimates over all search traffic. The packet also preserves the underlying stratum sizes for reporting. The distributed repository contains code and aggregate manifests only. Reviewer text, case mappings and individual assessments stay local.

## Verified replay

Slurm job 42702 on ws5 reproduced the citation coverage and known-positive citation counts from job 42694. The deterministic local packets contain the following cases.

| Dataset | Top-five slots | Both nonempty and different | Only one arm returns evidence | Sampled candidate slots | Unique citations for review |
| --- | ---: | ---: | ---: | ---: | ---: |
| KDD | 610 | 20 | 15 | 32 | 82 |
| SciRepEval | 295 | 2 | 7 | 17 | 36 |

All 118 citation cases have at least two additional profile-title anchors. These anchors still require external identity checks. The blank submissions pass the integrity check and have zero independent judgments.

The number of changed packages also bounds the possible benefit before reviewing any citation. Taking every changed nonempty package in the most favorable direction and accounting for one-sided output, the full-panel clean-citation P@5 difference is bounded by **−4.75 to +4.26 points** on KDD and **−1.36 to +2.37 points** on SciRepEval. These are conservative finite-panel limits under fixed rankings and outputs. They are not observed quality effects or confidence intervals. The pending review can resolve how those changes affect citation quality. [Aggregate replay and blank-review validation](summary.json)

## Reproduce

Use the pinned CoF snapshots and the two result files from the shared-paper experiment. Generate each packet inside a Slurm CUDA allocation.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.cof_citation_package_review export \
  DATASET OWNERSHIP_RESULTS EMBEDDING RERANKER NEW_PACKET
python -m experiments.retrieval.cof_citation_package_review compare \
  PACKET REVIEWER_A REVIEWER_B NEW_AGGREGATE
```
