# Further Review of the Closest Related Work

## Industry Expert Semantic Tree (Feng et al., 2026)

Local full text: docs/research/2026_MDPI_Bridging the Semantic Gap in Industry Academia Collaboration A Two-Stage RAG System for Intelligent Expert Recommendation.txt; DOI: https://doi.org/10.3390/mca31030102. The methods and experiments sections have been read.

Its method includes offline domain/keyword extraction, semantic normalization, a domain–concept–outcome–expert hierarchy, online LLM intent parsing, semantic matching, and peak-outcome aggregation, and provides a Top-3 average aggregation comparison. The experiment scale is 100 experts, 999 papers, 1,000 patents, and 20 manually annotated real-world needs. The original reports P@5=0.784, R@10=0.812, and an average retrieval latency of 115.8ms.

This is very close to this project’s task of “industry terminology → academic capabilities → expert evidence.” Using only a semantic tree, offline LLMs, peak-outcome aggregation, or Top-3 aggregation cannot constitute an independent contribution. Effective designs should be retained and this work included as an in-domain comparison. Latency comparisons require consistent timing boundaries for online intent parsing, network encoding, and explanation generation; figures from papers with different hardware and dataset sizes cannot be compared directly.

Verifiable directions for differentiation: source-constrained synthetic tasks and counterfactual repair, necessary conditions and unknown states, sourced capability cards that do not require per-query generation, and multi-resolution recall with bounded incremental evidence determination. Whether these provide gains remains subject to experimental validation.

## LiteRAG (2026 preprint)

Full text: https://arxiv.org/html/2609.10239v1. The methods section shows that it combines semantic, lexical, and community-anchor signals; expands subgraphs using query-adaptive thresholds and hub penalties; sets limits on depth and expansion count; and ultimately constructs compact relational-chain context. It has already moved the model out of the retrieval control loop.

Therefore, this project should borrow its algorithmic retrieval and compact-evidence ideas, rather than claim that “offline organization and little online generation” is a new concept. The differences should lie in conditional combinations of expert capabilities and evidence states, the data-synthesis review loop, and quality–time validation. This paper primarily studies graph question answering, so its reported gains cannot be directly transferred to expert ranking.

## Implications for the Current Experiments

- Retain dense, BM25, profile aggregation, and sourced summaries as baselines or ablation components, respectively.
- Graph indexing, early stopping, batching, and cached references are already established techniques; do not remove them merely because they already exist.
- Center the research conclusions on the actual contributions of the full combination and the data/protocol, avoiding presenting a change in algorithm name as a new method.
- Limitations on in-domain labels (CMU, etc.) and industry tasks must be reported; SciFact validates only the retrieval component.
# Supplementary Review of Nearby Work on Constraints and Scope Evaluation

Three original works directly related to the current data/metrics were additionally checked:

- FollowIR (NAACL 2025, https://arxiv.org/html/2403.15246v3) uses paired, modified retrieval instructions and p-MRR to check whether rankings respond to changes in conditions. Our single-condition counterfactual is not a new evaluation idea; this baseline should be retained, and ranking responsiveness should be distinguished from the correctness of evidence support.
- ComLQ (AAAI 2026, https://arxiv.org/html/2511.12004v2) constructs queries with logical structure and uses LSNC to check negation constraints. Using LLMs with logical structures to synthesize complex retrieval tasks, and adding constraint-violation metrics, both have clear prior work; we cannot claim to be the first. Its generated data are expert-validated; our current model review does not offer an equivalent guarantee. The official code/data entry point is https://github.com/xgl-git/ComLQR-main; it has not yet been integrated into local retrieval experiments.
- n-Clue (2026 preprint, https://arxiv.org/abs/2607.24165, official repository https://github.com/sunggukcha/n_clue) evaluates the ranking of complete-constraint evidence relative to partial-constraint matches, directly addressing the motivation that “satisfying only some conditions should not be compensated for by a high score.” Our mandatory gate needs to be compared with its complete-first / n-Clue Score; AND gating itself cannot be treated as a new theory. The current repository is preparing to download it for independent reproduction; frozen leaderboard results have not been treated as results for our method.

Potentially distinctive aspects of the setting are: capability evidence from multiple sources for the same expert, the scope of relationships between independent capabilities and joint applications, the distinction between missing information and an actual lack of capability, and evidence completeness under a bounded end-to-end time. These remain directions for combined research that require validation, not original contributions that have been proven.

Follow-up empirical update: All official declarations and file checksums for n-Clue’s fixed commit 8a19258 passed; ComLQ’s fixed commit 93d3219, with 11,251 documents and 2,909 queries, has been integrated and BM25/TF-IDF runs have been completed. The summary is in comlq-lexical-results.json. The former is a reproduction of a frozen leaderboard; the latter is a newly run local lexical baseline. They must not be conflated with results for our model method.


## Further Boundaries for Query-Association and Annotation Budgets

[Diaz et al. (CIKM 2020)](https://www.microsoft.com/en-us/research/publication/evaluating-stochastic-rankings-with-expected-exposure/) study expected exposure and fairness under repeated queries. [Boratto et al. (IPM 2021)](https://arxiv.org/abs/2006.04275) propose popularity-related evaluation based on item recommendation probabilities and true-positive rates. The current cross-query exposure–relevance association is not equivalent to either definition, but it is not evidence of “being the first to eliminate popularity effects.”

[Schnabel et al. (ICTIR 2016)](https://www.microsoft.com/en-us/research/publication/unbiased-comparative-evaluation-ranking-functions/) formulate system comparison under missing judgments as a Monte Carlo estimate, providing an unbiased estimator and an optimized sampling distribution. Our deterministic box intervals and querying by priority of absolute coefficient address the worst-case width for fixed unknown labels; they cannot be presented as a new unbiased sampling theory, nor can identification intervals be called confidence intervals. Replay shows that the annotation cost of lift may be far higher than that of ordinary precision differences; currently, there is no validated metric contribution.

## Classic Expert Models That Must Be Retained

[Balog, Azzopardi, de Rijke (SIGIR 2006)](https://doi.org/10.1145/1148170.1148181) marginalize at the candidate-term distribution and document-query likelihood levels, respectively. We implemented Model 1 with document-normalized association, and Model 2 with both document-centered and candidate-centered association, fixing λ=.5 with no label-based tuning. The candidate-term distribution is the average of normalized document probabilities; it must not be incorrectly implemented as concatenated term frequencies. Precompiled additive term–expert contributions are an engineering optimization; the semantic ranking itself remains a classic model. The NIPS results are a reminder that subsequent methods must not compare only against BM25 and weak profile fusion.
