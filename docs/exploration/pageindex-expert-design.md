# PageIndex to Expert Evidence Index: Design and Validation Plan

Review target: VectifyAI/PageIndex, commit `c9f1a879d3b3906ec2617724a905f5cb5e32546e`. Source: https://github.com/VectifyAI/PageIndex. The local research copy is at `artifacts/third-party/PageIndex`; the upstream implementation has not been merged into this project.

The current official implementation separates structure extraction, summarization, and agent retrieval; tree nodes contain titles, summaries, and original-text locations. `tree_optimize.py` further determines whether to merge or expand based on routing cost and scanning cost, rather than assuming that a deeper tree is better. Its page-level cost model cannot be applied directly to multi-source expert materials; token usage, call counts, and end-to-end latency need to be measured. The README’s financial question-answering results and the performance of the commercial retrieval interface do not represent results on expert tasks, nor can they automatically be attributed to open-source tree navigation.

## Gaps in the Existing Implementation

The existing `ExpertAgent` already has a shallow “expert → evidence type/field → leaf” tree, but it traverses the entire docs table online, takes only the first 40 records directly from each branch, and then has the model select at most 30 records. It has no offline capability-semantic nodes, no global reduction of the expert search scope, and no evidence that routing preserves the required evidence. Merely naming this structure PageIndex does not constitute an improvement.

## Proposed Two-Level Coarse-to-Fine Index

1. Global routing directory: technical problem/method/object → candidate experts; also preserve source attribution and cross-links across domains, avoiding the forced assignment of an expert to only one domain. Use precomputed lexical or frozen-vector retrieval at this layer for fast candidate retrieval, rather than first performing serial LLM tree search over the entire corpus.
2. Expert evidence tree: expert → problem and method capability nodes → actual papers/profiles → original-text spans. Summaries are for locating evidence; leaf-level original text is for assessment. Nodes record facts already demonstrated, sources, and gaps; they do not present potential transfer as a completed project.
3. Requirement-driven detailed search: divide enterprise requirements into fidelity conditions, and for each candidate expand only capability branches that could change the ranking, assessing evidence and applicability in batches. If a coarse summary contains no information, retain an original-text fallback or multi-path exploration; an expert must not be permanently excluded merely because the summary does not mention something.
4. Output: expert ranking, evidence of direct application, explainable method transfer and its assumptions, and unmet requirements. High-confidence facts and future problem-solving capabilities are represented separately.

## Combined Contribution to Validate

A potential contribution to study is combining a source-traceable expert capability directory, requirement-scope fidelity, budgeted expansion order, and applicability ranking to reduce irrelevant experts and unnecessary original-text reading. Tree indexing, hierarchical search, offline summarization, and upper-bound pruning all have prior work and must not be claimed as individually novel. Whether this is useful will be determined by experiments.

At minimum, comparisons should include: reading all evidence; flat frozen embedding + fixed top evidence; the existing shallow evidence tree; an expert semantic tree; and an expert semantic tree with global coarse routing. Keep candidate scale, model, and token budget comparable. Record expert recall, required-evidence recall, final applicability ranking, citation support, validity of transfer rationale, and the time spent in each online stage: encoding, routing, reading, and assessment. An exact-score upper bound can guarantee only the score as defined; it cannot guarantee retention of potentially true experts or the quality of semantic reasoning.

First build reviewable nodes and fixed baselines for a set of experts with abundant paper evidence in real data, then expand to all 10673 people. Both stages must report experts with missing information; they must not be treated as negative examples by default.

## First Change Implemented

Added `ExpertCatalog`, which creates an expert-attribution and file-offset index for all original evidence during the offline stage, bound to the corpus SHA256. When the existing ExpertAgent constructs a candidate tree, it now reads only sources belonging to candidate experts, removing the step of parsing the entire corpus’s original text on every query; other routing and ranking logic remain unchanged for now. In three alternating-order comparisons involving 21 experts and 296 sources, the content and order were exactly the same as with a full scan; scanning took about 347ms, while candidate reading had a median of about 1.44ms (about 7.05ms on the first run). This is an engineering optimization for source reading, not evidence of end-to-end agent acceleration or semantic-tree accuracy. All 87 existing tests passed. The next stage still requires genuine problem/method capability branches and testing coarse-routing recall.

## Cross-Source Navigation Evidence

The two HotpotQA runs use one shared index of 14,950 canonical passages from 1,536 development questions. Every arm fixes the first dense result and limits the two returned passages to 1,600 characters. Success requires both annotated sources. The same success counts held when requiring all exact normalized supporting sentences.

| Held-out result | Flat dense | Follow title link when available | Comparison cue rule | Learned switch |
| --- | ---: | ---: | ---: | ---: |
| All questions | 172 / 512 | 283 / 512 | 288 / 512 | 291 / 512 |
| Bridge | 83 / 404 | 214 / 404 | 203 / 404 | 206 / 404 |
| Comparison | 89 / 108 | 69 / 108 | 85 / 108 | 85 / 108 |
| Request P95 (ms) | 7.151 | 7.152 | 7.158 | 7.159 |

- **Replication.** Job 40183 returned both sources for 202 of 512 queries with flat ranking and 286 with links. The graph helped bridge queries and hurt comparisons. Job 40189 reproduced the direction on new questions with normalized support titles and questions disjoint from the fitting cohort. The fixed switch fits 198 differing actions from the earlier 512 queries.
- **Controls.** An equal-degree random graph averaged 104 successes. A random switch at the same call rate averaged 259.3. The learned switch improved by 3 hits over the comparison cue rule. Its paper-cluster paired 95% interval was -0.98 to +2.16 percentage points. The rule is a strong baseline.
- **Publication relevance.** The supported claim is that article-title links help this benchmark under the tested budget and hurt comparison cases when used indiscriminately. The shared corpus includes benchmark contexts. Expert constraints, contributor attribution, action provenance and benefit over budget-matched strong retrieval need separate evidence. Graph expansion and context-conditioned retrieval have published precedents. [PullNet](https://aclanthology.org/D19-1242/) · [Multi-Hop Dense Retrieval](https://openreview.net/forum?id=EMHoBG0avc1)
- **Reproduction.** Run [shared source navigation](../../experiments/hotpotqa/shared_source_navigation.py) and [the held-out switch](../../experiments/hotpotqa/learned_source_navigation.py) on the pinned distractor development input. Both jobs ran on one RTX 5090 on ws5. Store reports under `artifacts/hotpot-shared-source-navigation-v1` and `artifacts/hotpot-learned-source-navigation-v1`. The latter contains a separate paired cue-rule comparison derived from the frozen rows.

## Expert-Ranking Check of Source Complementarity

Slurm job 40195 tested one RTX 4090 on ws6. Each arm reranked the same 20 experts from exact dense retrieval and returned up to two owned 500-character excerpts per expert. Fragment methods scored the title and up to two abstract sentences using frozen Qwen. The bottleneck methods maximized the weakest fragment score in one source or across two sources.

| Method | SIGIR hard P@5 | SIGIR P95 (ms) | SciRepEval known hard P@5 | SciRepEval P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Exact dense | 20.27% | 13.36 | 15.38% | 14.53 |
| Fragment mean | 20.00% | 28.81 | 16.48% | 29.85 |
| Single-source bottleneck | 20.82% | 28.75 | 14.73% | 29.78 |
| Two-source bottleneck | 20.55% | 28.93 | 15.60% | 30.00 |

- SIGIR used 73 queries and fully judged returns. The paired two-source versus single-source hard P@5 change was -0.27 percentage points, with a descriptive 95% interval from -1.92 to +1.37.
- SciRepEval used 91 queries. About 42% of returned positions had labels. The two-source versus single-source hard P@5 difference could range from -7.69 to +9.01 points under the missing labels. A judged-only shortlist analysis is conditional on the initial candidate pool.
- Exact dense remains the default for ranking. These experiments leave open selective fragment scoring and source verification. The reviewer grades score experts, and the heuristic fragments have no separate hard-constraint gold.
- Reproduction uses [the source-pair benchmark](../../experiments/retrieval/cof_complementary_source_pairs.py) and the pinned CoF vector indices. Output is under `artifacts/cof-source-pairs-v1`. The paired metrics, character budgets, and candidate-source ownership were checked against the original records.
