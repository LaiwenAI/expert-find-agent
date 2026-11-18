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

## Selective Fragment Scoring

Job 40200 ran on one RTX 4090 on ws6. The learned gate used eight initial-search features, including score margins, query length and source count. It fit 111 identified hard-label differences from 243 SIGIR and KDD queries. Of those, 13 favored fragmentation and 13 favored the default. Exact query IDs and normalized titles were disjoint between calibration and the NIPS and SciRepEval replays. Profiles could overlap.

| Venue | Frozen gate calls | Known hard P@5 difference vs. dense | Missing-label bounds | P95 dense / learned / always (ms) |
| --- | ---: | ---: | ---: | ---: |
| NIPS | 14 / 34 | -1.18 pp | -5.29 to +4.71 pp | 14.05 / 28.03 / 29.36 |
| SciRepEval | 29 / 91 | -0.44 pp | -3.08 to +3.52 pp | 14.69 / 29.32 / 30.04 |

The fixed threshold targeted 25% in calibration. It reached 41% and 32% in the evaluation venues. At matched call counts, the learned rule had +0.59 and -1.32 known hard P@5 points relative to the simple score margin rule. Both paired missing-label bounds span zero. The learned policy offers no stable advantage here. It also retains most of the always-on P95, despite reducing mean latency.

The [benchmark](../../experiments/retrieval/cof_fragment_gate.py) measures branch costs and replays the frozen gate. It does not run an independent HTTP service. Timing, candidate outcomes, and report hashes are under `artifacts/cof-fragment-gate-v1`. Reproduction checked the venue split, all route counts, and label bounds.

## QASPER Human-Annotated Evidence and Reader Cost

[QASPER (NAACL 2021)](https://aclanthology.org/2021.naacl-main.365/) supplies human-selected evidence paragraphs. The known-paper test isolates evidence acquisition after a source is selected. It does not assess expert identity, contribution attribution or screening across a global corpus.

Frozen Qwen3-Embedding-0.6B scores source paragraphs exactly within each paper. After two paragraphs, the policies stop, extend down the dense order, or prioritize unread sections. The learned policy predicts incremental evidence gain, weighting completion of any one annotator’s set above partial recall. It uses score margins, source structure, and initial query-term coverage. A deterministic paper split of the official development set supplied 401 fitting and 525 calibration questions. Both gates targeted 90% of the fixed-five context budget on calibration. No exact paper IDs, normalized titles, or complete body texts are shared between development and test.

| Policy | Official evidence F1 | Entire evidence set recovered | Context tokens mean / P95 |
| --- | ---: | ---: | ---: |
| Fixed two | 32.81% | 44.35% | 221 / 445 |
| Fixed five | 26.05% | 65.21% | 558 / 978 |
| Fixed eight | 21.53% | 75.40% | 894 / 1,529 |
| Score-gap gate | 30.17% | 60.11% | 512 / 1,275 |
| Learned gate | 29.07% | 58.92% | 516 / 1,299 |

Official text-only evidence F1 and tokens include all 1,451 test questions. Complete evidence means recovering every text paragraph from at least one nonempty, fully alignable answer annotation. This measure covers 1,256 questions across 402 papers and excludes image evidence. The learned gate's complete-evidence difference from fixed five is -6.29 percentage points (95% paper-bootstrap interval -8.50 to -4.08). It spends more tokens than the score-gap gate and offers no improvement in complete annotation coverage. Reporting F1 alone would favor fixed two, even though it loses 20.86 points in complete coverage. Both adaptive policies reduce mean tokens and increase P95. The shared live encoder, exact scoring, routing and comparison-context materialization took 7.23 ms at P95 on the RTX 5090 in job `40230`. This timing is shared across arms and cannot establish reader or service speedup.

A separately locked 72-question sample, one hash-selected question per paper, replays fixed five and the learned route through the same local BF16 Qwen3-4B-Instruct-2507 reader. Job `40235` alternates arm order; mean reader time falls from 184 to 162 ms (paired interval -53 to +8 ms), while P95 is 475 versus 478 ms. Answer F1 is 46.29% versus 43.69% (paired difference -2.60 points, 95% interval -8.49 to +3.67). Mean prompt tokens are 644 versus 597. Greedy generation is capped at 48 tokens and hits the cap in 11 versus 8 requests. The timed reader includes prompt construction through decoding, starting from already selected contexts. The sample does not establish a latency or answer-quality advantage. These results motivate a bounded expansion policy that measures complete evidence alongside F1 and caps reader-tail cost.

Reproduction: extract the official [development](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz) and [test and evaluator](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz) assets into `artifacts/public/qasper-v0.3`. Run [adaptive evidence](../../experiments/qasper/adaptive_evidence.py) and [paired reader](../../experiments/qasper/paired_reader.py) on a CUDA node with the pinned open-weight checkpoints. Source and evaluator hashes are verified by the runners. Local outputs are `artifacts/qasper-adaptive-evidence-v1` and `artifacts/qasper-paired-reader-v1`; source text and model outputs are excluded from the release package.

## Shared Sources With Separate Expert Checks

[TREC 2006](https://trec.nist.gov/data/t15_enterprise.html) provides separate expert and supporting-document judgments. A replay holds the ten initial candidates fixed using a leave-topic-out prior and hides each document grade until its expert-specific check. Each source read can be reused by ID across candidates. The source-sharing policy favors documents connected to several unresolved candidates. The controls are round robin, cache-first round robin, and a source-sharing replay whose individual check cap is inherited from the cache-first arm.

| Policy at cap 10 reads / 20 checks | Mean reads | Mean checks | Supported P@5 |
| --- | ---: | ---: | ---: |
| Round robin | 9.26 | 12.99 | 20.95% |
| Cache first | 9.14 | 13.12 | 19.86% |
| Share by source | 6.39 | 17.24 | 23.74% |
| Share by source, matched check cap | 4.65 | 13.10 | 21.26% |

Across 49 topics and 12 fixed source-order draws per topic, the independent-noise profile assumes 90% sensitivity and 1% false-positive rate. With the same caps, sharing gains 3.88 percentage points over cache first (topic-bootstrap 95% interval +0.68 to +7.62) while making more checks. Compared with round robin, the point gain is 2.79 points. With the matched check cap, the cache-first difference falls to +1.39 points (-1.60 to +4.86). The unconstrained sharing arm also increases unsupported promotions from 4.15% to 6.09%. Under linear costs, it is cheaper than cache first only when a document read costs more than 1.50 expert checks. Shared-cost simulation results are not service timings or measured verifier error rates. The pool already exposes judged candidate-source links and the matched cap is a comparison control.

The source-sharing opportunity also varies in an actual retrieval workload. Among the full exact CoF top-five lists from the [shortlist benchmark](../merit-suitability-card.md#global-shortlist-validation-on-cof), there were repeated source IDs in 49/91 SciRepEval queries and 0/34 NIPS queries. Deduplicating SciRepEval source IDs within each query reduces 1,365 expert/source links to 1,244 document reads (8.9% fewer). Ownership checks still apply to each link.

A live Qwen3-4B-Instruct-2507 test compares separate forwards, left-padded full batching, and prefix KV reuse followed by separate expert suffixes. It selects one shared source in each of 48 SciRepEval queries with overlap. Across the 109 checks, the common query-and-source prefix averaged 393 tokens. Inputs are token-identical across arms, with correct position IDs in the batch arm.

| Verifier execution | Median / P95 group time | Choices matching separate forwards |
| --- | ---: | ---: |
| Separate forwards | 56.08 / 94.76 ms | 109 / 109 |
| Full batch | 53.82 / 91.43 ms | 106 / 109 |
| Shared prefix | 49.77 / 65.09 ms | 108 / 109 |

On RTX 5090, mean group time falls by 6.72 ms relative to full batching (paired 95% interval -9.40 to -4.21). This is a forward-pass measurement on shared-source requests only. One BF16 decision changes relative to separate forwards, and no human labels establish whether either choice is valid. A fallback for near-tied decisions needs separate evaluation before adopting reuse. Distinct suffixes and per-expert judgments are retained.

Reproduce using [the replay](../../experiments/trec_enterprise/shared_source_budget.py) and [the live prefix comparison](../../experiments/retrieval/cof_shared_prefix_reader.py). Jobs `40245` and `40251` used ws5. Local records are in `artifacts/trec-shared-source-budget-v2` and `artifacts/cof-shared-prefix-reader-v1`.

### Guarded Verification on Mixed Requests

Job `40259` reruns query encoding, global exact expert retrieval, owned citation materialization and local verification on every measured request. A fixed greedy source assignment is shared across arms. Full batching and source-prefix reuse use the same prompt tokens. Both guarded arms recheck low-margin choices with the serial reference at a fixed 1.125-logit margin. The threshold was frozen from the exploratory BF16 logit scale before this run. KDD and SciRepEval have disjoint released query IDs, normalized query titles and profile paper IDs.

| Collection | Shared queries | P95 ordinary batch | P95 guarded batch | P95 guarded reuse | Guarded reuse match |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciRepEval | 49 / 91 | 216 ms | 346 ms | 297 ms | 455 / 455 |
| KDD | 138 / 170 | 222 ms | 382 ms | 353 ms | 850 / 850 |

Guarded reuse saves 15.55 ms mean on SciRepEval (paired 95% interval -21.02 to -10.86) and 41.40 ms on KDD (-46.57 to -36.38) relative to guarded batching. Guarded batching matches all serial decisions too. Ordinary batching differs in 5 and 15 choices. Reuse calls serial fallback for 218/455 and 346/850 checks. This validates mechanical reproducibility relative to a reference on the observed queries. The result does not establish that the reference decisions are semantically correct or that the heuristic margin certifies parity. Timing covers model inference and in-memory retrieval, excluding HTTP, cold startup and external source I/O.

The optimization has established systems precedents. Assess any research contribution through expert-specific assignment, completeness and evidence-quality constraints beyond reuse itself. [SGLang (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html) · [CacheBlend (EuroSys 2025)](https://doi.org/10.1145/3689031.3696098)

Run [guarded shared requests](../../experiments/retrieval/cof_guarded_shared_requests.py) on a CUDA node with the CoF frozen indices and the same Qwen checkpoints as the two preceding experiments. Local output is `artifacts/cof-guarded-shared-requests-v1`. The [PLOS role pilot](../../benchmarks/plos-credit-pilot-v1/README.md) supplies a separate attribution target from publisher-declared contributions.

## Extracting Explicit Constraint Values

[IFEval](https://github.com/google-research/google-research/tree/7117df5f368e1301e34bd014be9c93e24a4e73c6/instruction_following_eval) provides labeled response constraints. The diagnostic scores a four-kind projection of its prompts: word bounds, sentence bounds, forbidden terms and comma bans. It selects 72 prompts with a target annotation and 24 without one by a fixed ID hash. Exact success requires the entire projected set, including each numeric value and bound direction. This is a parser diagnostic on response instructions. Enterprise search intent and general instruction following need different evaluations.

A first Qwen3-4B JSON generation trial passed structure checks on 12/96 full prompts. An interface revision passed on 2/12 disjoint format-check prompts. The bounded experiment replaces open generation with fixed choices over values proposed from the source. Full and localized readers receive the same candidates. The localized reader sees cue-bearing sentences and adjacent text. The hybrid accepts directly parsed literal counts and comma bans when each tracked cue is covered, otherwise it runs the full candidate reader.

| Route | Exact projected set / 96 | Prompts with extra constraints | Reader calls | Request P50 / P95 |
| --- | ---: | ---: | ---: | ---: |
| Full candidate verification | 49 | 41 | 75 | 14.87 / 111.68 ms |
| Localized candidate verification | 48 | 39 | 75 | 14.66 / 117.16 ms |
| Literal rules, then full verification | 55 | 35 | 50 | 11.10 / 111.73 ms |

The hybrid handles 25 source prompts directly, all matching the projected labels on this sample. It gains 6.25 percentage points in exact set recovery over full verification (paired request-bootstrap 95% interval +2.08 to +11.46). This interval describes the sampled prompts. Template relationships can reduce effective independence. The reduction in reader calls improves median latency while P95 is essentially unchanged. The localized route shortens 38 prompts and matches only 21/43 forbidden terms, versus 26/43 with full context. Proposal coverage already omits 13/43 forbidden terms, 2/21 word bounds and 3/21 sentence bounds. Even a source-derived candidate can be assigned to an unintended scope. Extra constraints remain a failure mode in 35 hybrid outputs.

Run [the bounded extractor](../../experiments/ifeval/anchored_constraint_extraction.py) with the pinned IFEval input file and open Qwen checkpoint. Slurm job `40334` measures all three live arms in rotating order on an RTX 5090. Timing covers warm candidate generation and inference. Local records are in `artifacts/ifeval-bounded-constraints-v1`.
