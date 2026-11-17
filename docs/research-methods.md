# Problem Definition and Methodological Basis

Historical research entry point; the current task definition, evidence boundaries, and latest results are governed by the [SIGIR English LaTeX main manuscript](paper/sigir/main.tex). Insufficient data is not treated as a negative label for actual capability.
## Task

Given an enterprise technical need q and a set of experts E, where each expert has evidence D_e such as research areas, a profile, projects, and associated papers, output an ordered list of experts with verifiable evidence. The goal is the degree of technical match supported by evidence; this is not equivalent to academic prestige, paper count, or willingness to be contacted. The available data cannot verify experts’ actual availability, willingness to collaborate, or ability to deliver projects.

## Local Paper Reading

Prioritize reading the papers that actually exist in `/Users/zhoudixin/Desktop/Lai/Benchmark/literatures`. Full-text extractions of three papers are stored in docs/research; not all papers in the directory have been read in full, one by one.

1. Balog, Fang, de Rijke, Serdyukov, Si. *Expertise Retrieval*. Foundations and Trends in Information Retrieval, 2012, DOI 10.1561/1500000024. Section 3 distinguishes expert finding from expert profiling; Section 4 discusses evaluation metrics and test collections; Section 5 distinguishes candidate generation models, document generation models, voting, and graph methods. The current implementation uses an engineering baseline that aggregates experts after document retrieval; it does not strictly reproduce generative probability models.
2. Feng, Yang, Fang. *Bridging the Semantic Gap in Industry–Academia Collaboration: A Two-Stage RAG System for Intelligent Expert Recommendation*, 2026. The local full-text abstract proposes multi-source information, terminology normalization, a semantic tree, and coarse-to-fine retrieval; it is directly relevant to the current task. The paper’s vector-clustering-based semantic normalization has not been implemented in this project, so the current tree navigation cannot be described as a reproduction of that paper.
3. Sun et al. *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents*. EMNLP 2023. It uses generative models for relevance reranking and discusses permutation distillation. The current agent ranks evidence after checking it and includes citations; it does not implement the RankGPT sliding-window permutation protocol or distillation.

## Adapting PageIndex

Reference the README at https://github.com/VectifyAI/PageIndex: organize documents into a hierarchical index, then have a model search the tree and locate sources. The original project focuses on long-document question answering, whereas this project focuses on ranking multiple experts. The implementation first uses lexical retrieval to narrow the expert set, then constructs a tree of “expert → area/profile/papers → evidence leaf”; the model selects leaves and reads their contents, and finally produces an evidence-backed ranking.

This preserves tree navigation and source traceability; it does not implement model traversal of the tree independently of full-corpus vectors, nor does it directly call the PageIndex package. Omissions during candidate retrieval cannot be corrected by subsequent ranking. Truncating the display to the first 40 entries per branch is a current limitation; query-relevant leaf summaries or paginated navigation should be used later.

## Meaning of the Experiments

Recovering known associated experts from paper titles can be run without additional manual annotation; however, author attribution is only a proxy label, and other experts working on the same topic may also be relevant. The metrics answer “Can the known authors be recovered after excluding the evidence corresponding to the title?” They do not answer “Which expert should an enterprise choose?” Therefore, real-need experiments require separate annotations, and proxy accuracy must not be transferred to enterprise needs.

The current baselines are strongest-evidence BM25, voting over the top five BM25 evidence items, and TF-IDF variants. Dense-vector retrieval, hybrid fusion, and LLM reranking over the same candidate set still need to be added to determine the gains of tree navigation relative to common modern retrieval workflows. Experiments must compare the same query subset, exclusion rules, and candidate budget, and report both speed and model usage.
