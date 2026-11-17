# Top-Conference Literature Review and Boundaries of the Current Approach (2026-11-17)

Scope: Prioritize reading the full texts of ScIRGen, MERIT, and CaRR locally, and supplement them with ETS and HippoRAG 2. The full texts have been extracted, and the title pages and publication pages checked; this is not a claim that every paper in the literature catalog has been reviewed individually. Retain the existing valid index and citation checks; do not treat every component in the combination as original.

| Work and formal publication information | Conclusions directly relevant to this project | Treatment under the no-training constraint |
|---|---|---|
| [ScIRGen, KDD 2025](https://kdd.org/kdd2025/datasets-and-benchmarks-track-papers-2/), DOI 10.1145/3711896.3737432 | Task-oriented generation, alignment between materials and papers, hierarchical parsing, and quality filtering already have a systematic approach | Its data workflow can be borrowed; “real materials + model generation + review” alone cannot support a claim of being first |
| [MERIT, EMNLP 2026 Main Conference](https://aclanthology.org/2026.emnlp-main.2227/) | Separates key expert dimensions from incidental topical overlap, and uses a rubric to express requirements | Its requirements structure can be borrowed; its RL and distillation training are not part of this project’s implementation route |
| [CaRR, ACL 2026 Main Conference](https://aclanthology.org/2026.acl-long.950/) | Coverage, citation support, and evidence connectivity need to be checked separately | Its evaluation decomposition can be borrowed; atomic requirements or citation-chain checking themselves cannot be claimed as a new metric theory |
| [ETS, ACL 2025 Main Conference](https://aclanthology.org/2025.acl-long.1175/) | Treats evidence-set selection as a search with dependencies, rather than scoring each sentence independently | The original method trains strategies/value models; it cannot be described directly as a ready-to-use no-training implementation |
| [HippoRAG 2, ICML 2025](https://proceedings.mlr.press/v267/gutierrez25a.html) | Combines passages with structured chunks, and preserves context and fallback during retrieval | Can serve as a structured-retrieval comparison without in-house training; graph propagation cannot replace teacher attribution or factual support |

The local ScIRGen PDF is an arXiv version with a KDD publication page; the formal publication was separately verified against the KDD list. The local MERIT and CaRR PDFs include the formal ACL Anthology pages. The newly added ETS and HippoRAG 2 use PDFs from the official proceedings, rather than inferring publication status from filenames alone.

## Newly Added PDFs

Located at `/Users/zhoudixin/Desktop/Lai/Benchmark/literatures/method/`:

- `2025_ACL_Enhancing Retrieval-Augmented Generation via Evidence Tree Search.pdf`
- `2025_ICML_From RAG to Memory - Non-Parametric Continual Learning for Large Language Models.pdf`

The original texts, source URLs, and SHA256 values are stored separately in the literature catalog’s additions list. The user’s existing papers were not modified, and no preprint or journal article was renamed as a top-conference paper.

## Direct Implications for Existing Experiments

Current online applicability verification is both slow and unstable. The newly added EvidenceBench auxiliary development comparison tests only evidence selection within papers: 37 papers, 6310 sentences, and, when selecting 10 sentences, BM25/Qwen/MMR coverage rates of 0.3375/0.4770/0.4599. The top 30 Qwen sentences, independently reranked by Decisions, achieved 0.5619, with an average online time of 2.816 seconds; closed-set chat set selection over the same candidates achieved 0.5904, with an average request time of 2.040 seconds (35 valid outputs, 2 semantic fallbacks). The interval for the difference between the latter and the former crosses zero, and runtime was not randomly paired; no claim is made about interface superiority or gains to the complete system.

This exposes the distinction between set complementarity and item-wise relevance, supporting the inclusion of set selection in strong comparisons rather than equating independent probability ranking directly with sufficient evidence. This auxiliary data was generated and validated by an expert-guided upstream process; not all sentences were manually relabeled, and it is not enterprise-expert applicability gold data. The metric follows the upstream aspect-coverage definition and cannot be listed as our new metric.

## Next Falsifiable Experiment

Investigate demand-independent source-role indexing: distinguish, based only on the paper’s original text, the methods, subjects, findings, and limitations of the study; retain the original sentences and teacher attribution; and mark background and studies cited by others separately. On the query side, combine coarse retrieval with fallback to the original text; do not make hard judgments about expert capability directly from role prediction. This direction overlaps with existing knowledge compilation and contextual retrieval; innovation can only be established through combined effects in a specific setting.

First test coverage loss and online-call savings on a source-evidence test set not used to tune rules, while also accounting for offline compilation costs and conditions for reuse; then return to real enterprise requirements and teacher sources. If original-text evidence coverage cannot be maintained, do not replace the default path. Future goals, method transfer, and established facts must always remain separate; source-verification results cannot be upgraded into complete project capability labels.
