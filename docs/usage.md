# Usage and Reproduction

Python 3.11; corpus processing, BM25, TF-IDF, and the proxy benchmark depend only on the standard library. The model agent depends on `laiwenai==0.0.12` and reads configuration from the current working directory `.env`. Run all commands from the repository root.

```bash
PYTHONPATH=src python3 -m expert_search prepare
PYTHONPATH=src python3 -m expert_search index
PYTHONPATH=src python3 -m expert_search search $'\u673a\u5668\u4eba\u8fd0\u52a8\u63a7\u5236' --method bm25-max -k 5
PYTHONPATH=src python3 -m expert_search search $'\u673a\u5668\u4eba\u8fd0\u52a8\u63a7\u5236' --method bm25-vote -k 5
PYTHONPATH=src python3 -m expert_search search $'\u673a\u5668\u4eba\u8fd0\u52a8\u63a7\u5236' --method tfidf-max -k 5
PYTHONPATH=src python3 -m expert_search agent $'\u5bfb\u627e\u673a\u5668\u4eba\u8fd0\u52a8\u63a7\u5236\u548c\u5f3a\u5316\u5b66\u4e60\u65b9\u9762\u7684\u4e13\u5bb6' -k 3 --candidates 8
PYTHONPATH=src python3 -m expert_search benchmark
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Methods and Scope

- BM25-max: retrieves evidence first and takes the strongest evidence score for each expert.
- BM25-vote: sums each expert's five strongest evidence items. This is a truncated vote-style implementation; it does not claim to strictly reproduce Balog's generative model.
- TF-IDF-max: uses BM25's smoothed IDF, logarithmic TF, and square-root normalization by evidence length, then takes each expert's strongest evidence item. This is a TF-IDF variant, not a complete vector-cosine implementation.
- hybrid-evidence-tree: the model generates bilingual technical queries; reciprocal rank fusion combines retrieval results; a tree of "expert -> field/paper -> evidence leaf" is displayed for the model to select leaves and read their full text passages, and recommendations are then generated from the evidence. It adapts PageIndex's tree navigation and verification ideas; the first stage still uses BM25, so this is a hybrid method. Currently, at most 40 leaves per branch and 30 leaves in total are displayed, and each leaf is at most 6,000 characters long. These truncations may affect recall.

All citations are checked for evidence existence and expert attribution; this does not guarantee that the model's explanation of relevance is correct. The trace records model outputs and elapsed time for each stage, while usage records token usage returned by the model. The default command sends the requirements and candidates' research materials to the model service configured in `.env`.

## Validation to Date

`artifacts/agent-smoke.json` contains one real model run, returning three experts and evidence. This single case shows that the workflow runs; it does not establish recommendation quality. Proxy benchmark rankings and configuration are stored in `artifacts/benchmark/runs.jsonl` and `manifest.json`, respectively.

## Future Research Directions

The unified comparison now includes TF-IDF and the tree agent. Future work could extend this to dense retrieval, human evaluation of real requirements, recovery of structured model outputs, and ablation of tree-branch truncation. Recovering paper titles and authors is only a proxy task and is not equivalent to matching industrial expert requirements.

## Experiment and Audit Commands

```bash
PYTHONPATH=src python3 -m expert_search agent-benchmark
PYTHONPATH=src python3 -m expert_search demand-cases
PYTHONPATH=src python3 -m expert_search report
PYTHONPATH=src python3 -m expert_search verify
```

By default, agent-benchmark evaluates the first 10 baseline queries; it saves and can resume each result, and reports an error if the input manifest fingerprint does not match. report compares methods on the same completed-query subset; tables generated during a run are interim results. verify recalculates metrics and checks expert IDs, ranking deduplication, evidence sources, and holdout rules. demand-cases runs the first three sampled queries by default and saves the raw results from each baseline and agent for qualitative inspection only.

## Quick Research-Field Search Interface

```bash
PYTHONPATH=src python3 -m expert_search fast-agent $'\u673a\u5668\u4eba\u63a7\u5236\u4e0e\u5f3a\u5316\u5b66\u4e60' -k 5
PYTHONPATH=src python3 -m expert_search fast-agent $'\u5bfb\u627e\u7814\u7a76\u300c\u673a\u5668\u4eba\u300d\u548c\u300c\u4eba\u5de5\u667a\u80fd\u300d\u7684\u4e13\u5bb6' -k 5 --explain
```

Outputs NDJSON events: retrieval results are returned first, followed by model explanations when explain is enabled. First-batch time, index initialization time, and total explanation time are reported separately; first-batch latency must not be treated as the latency for a complete explanation. This entry point searches only Chinese research-field fields. An explicit "term" is matched jointly and exactly against public fields; ordinary text is ranked by field BM25. It is not equivalent to verified expert capabilities and does not cover all paper materials.

Add `--full-evidence`: for ordinary text searches, first output field results, then search all indexed profiles and paper evidence and output `full_evidence` events. Rankings may change; explanations use the final-stage results and do not cite candidates that have been replaced. Explicit "term" conditions retain exact research-field semantics and are not automatically overridden by paper similarity.

```bash
PYTHONPATH=src python3 -m expert_search fast-agent $'\u673a\u5668\u4eba\u8fd0\u52a8\u63a7\u5236\u4e0e\u5f3a\u5316\u5b66\u4e60' --full-evidence --explain
```

The full-corpus stage performs lexical evidence ranking; it is not yet validated semantic reasoning. Latency for first results and complete explanations must be reported separately.

`--reasoning-effort none` can explicitly disable additional model reasoning, only for model requests made when explanations are enabled. It does not change retrieval rules and is not written into the evidence input. The default remains unspecified pending a complete comparison to confirm quality.
## Logical Retrieval and Scope-Control Diagnostics

The official ComLQ archive is downloaded using a fixed commit and SHA-256; logical types are retained, and answer fields are not passed into retrieval queries:

```bash
PYTHONPATH=src python3 -m expert_search.research_cli comlq-lexical
```

This command runs BM25 and TF-IDF document-retrieval baselines and summarizes by logical type; it does not indicate industrial expert-matching quality. No explicit data license was found in the original repository, so the archive is not included in our source release package.

The official frozen n-Clue leaderboard can be independently recalculated:

```bash
python3 experiments/reproduce_nclue.py
```

This command verifies a fixed upstream version and recalculates only the published leaderboard. It does not run our retrieval methods or measure reasoning latency.

The scope-control synthetic experiment retains items not covered by the original requirement; the counterfactual changes only one topic in code:

```bash
PYTHONPATH=src python3 -c "from expert_search.synthesis_v2 import build; print(build(n_demands=20, output='artifacts/synthesis-scope-v5', protocol='scope-v5'))"
PYTHONPATH=src python3 -m expert_search.scoped_export
PYTHONPATH=src python3 -m expert_search.scoped_binding
```

The last two items export local silver-review candidates without source text and separately review joint application relationships. The samples are not a complete gold standard for industrial requirements; disagreements remain unresolved and are not assigned a definite relationship score.

## Direct Paper-to-Expert Matching Benchmarks

```bash
PYTHONPATH=src python3 -m expert_search.cof_data
PYTHONPATH=src python3 -m expert_search.research_cli cof-sigir-lexical
PYTHONPATH=src python3 -m expert_search.research_cli cof-sigir-dense
```

The first command downloads and verifies the official CoF data archive, reviews candidate annotations, and checks exact text copying. The next two run lexical and frozen-API vector comparisons over the complete judged candidate sets from SIGIR; labels are not included in encoder input. Vector results use precomputed queries, and scoring time is not equivalent to online encoding and retrieval. See `docs/exploration/cof-datasets.md` for data adaptation and license scope.

Paired evidence-budget analysis (reads completed records only; does not call models):

```bash
PYTHONPATH=src python3 -m expert_search.packing_analysis --dataset SIGIR --serialization inline
PYTHONPATH=src python3 -m expert_search.packing_analysis --dataset SIGIR --serialization shared
PYTHONPATH=src python3 -m expert_search.packing_analysis --dataset NIPS --serialization shared
```

The analyzer recalculates metrics on source labels and checks candidate pools, anonymized IDs, ordering, and failure fallbacks. If any planned record is missing, it outputs only completeness status rather than substituting the completed subset for a full comparison. Once complete, it reports paired-query bootstrap, mean/median/P95, failures, and rate-limit waiting. An interval spanning zero does not prove equivalence, and model-call time is not end-to-end latency.

New classic baselines and indexing experiments (no neural model training):

```bash
PYTHONPATH=src python3 -m expert_search.cof_language_benchmark --dataset SIGIR
PYTHONPATH=src python3 -m expert_search.cof_language_benchmark --dataset NIPS
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/compiled_lm_benchmark.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/teacher_compiled_lm.py
PYTHONPATH=src python3 experiments/judgment_budget.py
```

`CompiledCandidateLM.save(path, source_sha256)` stores a numeric inverted index without pickle; `CompiledCandidateLM.load(path, expected_source_sha256)` checks the snapshot, array hashes, ranges, and ordering. Precompilation changes only how the classic candidate language model is executed, not the scoring formula; real teacher timing provides no quality conclusion in the absence of labels.

To build the candidate-first evidence path after preparing the corpus and its normal lexical index, choose new output paths (the builder refuses to overwrite existing indexes):

```bash
PYTHONPATH=src python -m experiments.retrieval.build_candidate_pipeline \
  --corpus artifacts/corpus --source-database artifacts/search.sqlite \
  --language-index artifacts/candidate-lm-v1 --evidence-database artifacts/candidate-evidence-v1.sqlite
PYTHONPATH=src python -m experiments.retrieval.audit_compiled_index \
  artifacts/search.sqlite artifacts/candidate-evidence-v1.sqlite
```

The builder checks corpus-to-document identity and ownership; the separate audit checks compiled posting scores. Both run offline and can require substantial time and disk space. Instantiate `CandidateEvidencePipeline` with those same paths, then call `search(query, k=20, evidence_per_expert=1)` for a candidate shortlist with owned evidence. For model reasoning, pass that pipeline to `CandidateReasoningAgent(pipeline=...)` and call `run(query, k=5, candidates=20)`. The latter makes one provider call and requires configured `laiwenai` credentials. Citation ownership is checked, but neither source relevance nor individual contribution is automatically verified. This path uses a different candidate ranking from the basic BM25 CLI.

Local frozen encoding and synthetic-candidate preparation use the optional local environment:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-local/bin/python -m expert_search.cof_late_interaction --dataset SIGIR
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-local/bin/python -m expert_search.cof_late_interaction --dataset NIPS
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-local/bin/python -m expert_search.semantic_synthesis_pool
PYTHONPATH=src python3 -m expert_search.semantic_synthesis_pool --execute
```

The final command makes real model calls. Candidate preparation itself performs local retrieval only: it fixes twenty raw requirements, retrieves research fields by title, and adds the top three lexical evidence items and Chinese and English research fields for the same expert. It then still requires the complete scope-v5 queries and counterfactual review. Similarity is not used as a label. Since the source strategy also changes relative to the old version, an acceptance-rate change cannot be attributed to semantic retrieval alone. Run this type of generation task separately from network-timing experiments sensitive to rate limits.

Per-condition review and review-data export:

```bash
PYTHONPATH=src python3 -m expert_search.scoped_dataset_audit
PYTHONPATH=src python3 -m expert_search.scoped_dataset_audit --export
PYTHONPATH=src python3 -m expert_search.compiled_expert_search $'\u673a\u5668\u4eba\u8fd0\u52a8\u63a7\u5236\u4e0e\u8def\u5f84\u89c4\u5212' -k 5
```

The review handles only two batches of existing silver candidates. Checking condition IDs and unique citations does not amount to a human semantic gold standard. Export revalidates source files and raw model outputs; the output contains neither teacher/requirement text nor model traces. The complete synthetic review artifacts remain local candidates and are not authorized for redistribution as a third-party corpus.

Grouping and source-support diagnostics for review data:

```bash
PYTHONPATH=src python3 -m expert_search.scoped_dataset_audit --export
PYTHONPATH=src python3 -m expert_search.grouped_dataset
PYTHONPATH=src python3 -m expert_search.scoped_silver_benchmark --split test
```

Regenerate the split after re-export. The loader verifies hashes for split and pairs, as well as the prefix content, hash, and expert attribution for every complete input source. This diagnostic is not an assessment of expert relevance or actual capability; see `docs/scoped-dataset-card.md`.

### Frozen Cross-Encoder and Semantic Transfer Exploration

First prepare `packing-shared-v2` inputs using the existing CoF workflow. The first cross-encoder run downloads the model and requires optional dependencies; the model is not included in the source package.

```sh
python -m pip install -e '.[rerank]'
PYTHONPATH=src python -m expert_search.cof_crossencoder --dataset SIGIR --model ms-marco-TinyBERT-L-2-v2 --mode source-max
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/semantic_transport.py --dataset SIGIR
```

Semantic transfer also requires `.[local]`. The fixed protocol is located at `docs/exploration/semantic-transport-protocol.json`; the complete working directory retains this protocol. The two experiment types write to separate CoF artifact directories, and kernel and vector construction costs are excluded from online scoring time. The source-counterfactual correction review command is `python -m expert_search.source_counterfactual --reaudit`; it depends on original local v1 artifacts and does not constitute an independently downloadable dataset.

### Execution Optimization for Exact Ordered Top K

The persistent teacher-query entry point now uses vectorized upper bounds and exact replay, preserving the classic model's top-K ordering and returned scores, including ID tie-breaking. It still accesses every active inverted-list item and does not validate expert capabilities. Complete teacher-library paired re-evaluation:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/certified_topk_benchmark.py
PYTHONPATH=src python3 experiments/semantic_transport_analysis.py
```

The former depends on the existing teacher compiled index and 20 raw requirements, and writes to `artifacts/certified-topk-v2/`. The latter rechecks source metrics on both complete CoF query sets and calculates exploratory paired intervals; equal mean P@5 does not imply equal rankings.

The requirement-associated source-control workflow provides an explicit local preparation path:

```sh
PYTHONPATH=src python experiments/demand_aligned_source_controls.py --local
```

Install `.[local]`. On first use, it downloads a multilingual MiniLM ONNX export, with a maximum of 128 tokens. This step prepares sources only and produces no valid review labels. `--execute` runs the two-model review construction and still requires a gateway supporting GPT-4o and GPT-6-Luna in the protocol. The default API embedding path and `--local` use separate artifact directories; the backend is not silently replaced.

If the review backend is unavailable, generation-feasibility results can be retained separately: `PYTHONPATH=src python3 experiments/demand_aligned_source_controls.py --local --generate-only`. This command uses already-prepared local sources and does not require FastEmbed to be installed in the runtime environment, but it requires laiwenaI and an available GPT-4o. Output is written only to the generation-only subdirectory; it neither replaces the two-model review nor serves as a publishable label.


The frozen Qwen baseline for Apple Silicon requires `.[mlx]` to be installed. Run `experiments/prepare_qwen3_reranker.py` and `experiments/qwen_reranker_benchmark.py --dataset SIGIR` in order. The current entry point uses a memory limit of 4GiB and a cache of 128MiB; results are written to qwen-reranker-mlx-v2. `experiments/qwen_reranker_analysis.py` rechecks sources, candidate mapping, and per-query metrics; an incomplete run outputs completion status only.


The source-prioritized construction experiment entry point is `experiments/witness_first_controls.py`, which depends on prepared local-v4 sources and multilingual vectors. It retains responses from each stage: joint-relationship planning, independent source retrieval, and exact witness selection. `experiments/witness_control_auxiliary_audit.py` performs GPT-4o-mini-assisted review of all structurally successful samples from both construction workflows. The current four cases comprise three disagreements and one unresolved case; subsequent data export still requires checking support status and association with real requirements.

The Decisions adapter calls `client.decisions.create` with the new SDK; with the old SDK, it sends requests through the same native endpoint. The current gateway test returns 400; errors are preserved for the caller.


Run the prefix-cache paired experiment with `experiments/qwen_prefix_pilot.py`, using the first six protocol queries per dataset and two repetitions. Add `--fp32` to measure the FP32 model independently. The analysis entry point, `experiments/qwen_prefix_analysis.py`, uses the same options and requires all 24 records. Raw FP32 and BF16 records are stored in separate directories.


Generate the independent-review source-text package with `PYTHONPATH=src python3 experiments/build_blind_review.py`. The current output contains 8 review forms to complete, covering both source snapshots for 4 constructed cases. Review inputs use renumbered sources and hide model judgments, generator rationales, and the full/control condition names. The original-source mapping is stored separately in private-mapping.json; reviewers should read only inputs.jsonl and the instructions before submitting. The source-text package is stored in the artifacts directory; the current number of human labels is 0.


Validate a completed review file with `PYTHONPATH=src python3 experiments/validate_blind_review.py <submission.jsonl>`. Validation checks the input snapshot hash, project completeness, source and condition text, reviewer identity, labels for three conditions, and the unique locations of supporting citations and requirement anchors. The output retains the submission file hash for subsequent verification. Blank templates are rejected.


Run the fixed language-model checks for the larger CoF set with `experiments/cof_scale_check.py --dataset KDD`; SciRepEval is also supported. `cof_scale_analysis.py` recalculates separately under the original protocol and on the set excluding exact duplicate queries. After completing the lexical baseline, run `cof_scale_comparison.py` to compare BM25-max and BM25-top3 per query on P@5 and nDCG@10, and output exploratory paired intervals. If any planned query is missing, output completion status only. DOI query IDs use URL-encoded filenames.

After the Decisions API is restored, run real-source-view experiments using the configured `.env` and `gpt-6-luna`:

```sh
PYTHONPATH=src .venv-local/bin/python experiments/decisions_blind_pilot.py
PYTHONPATH=src .venv-local/bin/python experiments/decisions_batch_comparison.py
```

Both depend on local `artifacts/blind-source-review-v1/inputs.jsonl`. The second experiment randomly assigns the order of item-by-item and batched processing for the same view, and saves per-request responses, latency, and label consistency; model output does not constitute a human gold standard.

Fixed-ranking-fusion exploration (requires `scale-check-v1` results for KDD and SciRepEval, plus a local BM25 index):

```sh
PYTHONPATH=src python3 experiments/cof_rank_fusion.py
```

The fixed approach is an equal-weight reciprocal rank fusion of Model1 and BM25-top3, with constant 60. It saves complete candidate rankings, per-query metrics for both original methods and the fusion method, and paired bootstrap results. This exploration was proposed after inspecting baseline means and must not be labeled a confirmatory experiment on an unseen test set; it reuses cached Model1 rankings and does not report online latency.

Exploration of strongest word-level evidence aggregation:

```sh
PYTHONPATH=src python3 experiments/cof_max_term.py --dataset NIPS
PYTHONPATH=src python3 experiments/cof_max_term.py --dataset KDD
PYTHONPATH=src python3 experiments/cof_max_term.py --dataset SciRepEval
```

For each expert and each word, this approach precomputes the maximum BM25 contribution across their papers and sums these at query time. The results directory is `max-term-v1` for the corresponding dataset. It tests offline-compiled word-level aggregation; cross-paper word matches must not be interpreted as evidence of joint capability. Timing excludes index initialization.

Requirement-first data-construction preparation:

```sh
PYTHONPATH=src .venv-local/bin/python experiments/demand_first_facets.py
PYTHONPATH=src .venv-local/bin/python experiments/prepare_demand_first_pool.py
```

Wait for the first step to complete before running the second. Extraction results are in `artifacts/demand-first-facets-v2`; teacher evidence candidates driven by research topics are in `artifacts/demand-first-pool-v1`. These candidates have not undergone independent relevance or capability labeling and cannot be used directly as benchmark gold standards.

Matching experiment that fixes requirements and teacher candidates while changing paper-evidence selection:

```sh
PYTHONPATH=src .venv-local/bin/python experiments/demand_first_semantic_sources.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_demand_first_pool.py --semantic
```

First complete the original `audit_demand_first_pool.py` to freeze 13 trial tasks. The multilingual model is used only to rank the same teacher papers, using an existing local cache, with no model training. Model files, encoding inputs, and vector summaries are saved. The review output remains a single-model judgment and cannot serve as a final data label.

Select constructible candidates from complete real requirements and generate narrow capability conditions with paper citations:

```sh
PYTHONPATH=src python3 experiments/mine_demand_topic_overlap.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_topic_overlap.py
PYTHONPATH=src python3 experiments/prepare_paper_backed_overlap.py
PYTHONPATH=src .venv-local/bin/python experiments/generate_paper_backed_tasks.py
```

Complete each step before proceeding to the next. All stages retain rejected and unresolved results. The final step only generates conditions for review and cannot be used directly as dataset labels; local results contain private source text and are not distributed in the source review package.

Conduct blind two-model review of all successfully constructed paper capability conditions:

```sh
PYTHONPATH=src .venv-local/bin/python experiments/audit_paper_backed_tasks.py
```

Review inputs include the original requirement, the complete capability condition, and the frozen paper window, while hiding generator rationales, citations, and expected labels. GPT-4o and GPT-6-Luna must independently confirm requirement association and evidence support, and the citation must pass unique original-text validation, before an item can be labeled a two-model silver candidate for further processing. Both models belong to the same model family, so this still does not constitute a human gold standard.

Frozen Qwen3 Embedding 0.6B baseline (Apple Silicon MLX environment):

```sh
python3 experiments/prepare_qwen3_embedding.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-mlx/bin/python experiments/cof_qwen_embedding.py
```

Download the fixed official revision and verify the weight LFS hash; do not modify the weights. The encoding budget is fixed at 512 tokens, batch size 8, using the normalized FP32 vector of the last real token. Encoding blocks are stored in the corresponding CoF dataset's `qwen-embedding-v1`; changes to the input digest will prevent cache reuse. Ranking cached queries does not include online query encoding. During encoding, the GPU memory limit is 4 GiB and the cache limit is 128 MiB.

Prototype of a requirement-independent expert semantic evidence directory (fixed 21-position teacher feasibility queue):

```sh
PYTHONPATH=src .venv-local/bin/python experiments/build_expert_semantic_tree.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_expert_tree_nodes.py
```

Wait for the first step to complete before running the second. Each expert may have at most six "problem-method-object" nodes, which can cite only genuine source passages belonging to that expert. Structural and span validation of the constructed results does not establish that the summaries are accurate; the second step separately judges whether node descriptions are supported by their cited passages. Unsupported summaries should not be treated as verified capabilities; the original sources remain available as fallback.

Coarse-to-fine routing comparison over the full corpus:

```sh
PYTHONPATH=src python3 experiments/prepare_global_expert_roots.py
PYTHONPATH=src python3 experiments/prepare_global_expert_branches.py
```

Root nodes prioritize research fields and the titles of the first three papers; when these are missing, they retain the original-material fallback. The branch comparison retains a 1200-character prefix of every non-empty original source. Encoding scripts are `global_expert_root_benchmark.py` and `global_expert_branch_benchmark.py`, and require frozen Qwen3 Embedding and a CUDA environment. Slurm submission scripts are `cluster_global_roots.sbatch` and `cluster_global_branches.sbatch`. First transfer the input and code to the compute node's large disk, then submit from the login node. Source text serves only as a routing hint; neither prefix nor token truncation guarantees complete evidence coverage.


Same-node functional-representation diagnostic (prepare the local frozen model in advance):

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mlx/bin/python experiments/compare_functional_routes.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mlx/bin/python experiments/compare_functional_routes.py --condition-diagnostic
```

The second command uses a known silver condition only to distinguish a directory-representation problem from a full-requirement-mapping problem; it is not a deployment-quality measure. The two modes have independent input hashes, vectors, and result directories, and reject reuse of caches when inputs change.


Original-text applicability verification for directory candidates and separate model review (requires existing typed routing, requirement decomposition, and original local evidence; reads `.env` and calls gpt-6-luna and gpt-4o):

```sh
PYTHONPATH=src .venv-local/bin/python experiments/verify_typed_route_sources.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_source_applicability.py
```

Verification retains raw responses and rejects unknown attribution and citations that are not in the original text; structural validity proves only citation localization, not method applicability. Review results are saved separately; unresolved results are not treated as negatives, and none serve as evaluation gold labels.


Fixed passage-ID localization and separate component/full-item review:

```sh
PYTHONPATH=src .venv-local/bin/python experiments/verify_typed_route_sources.py --span-ids
PYTHONPATH=src .venv-local/bin/python experiments/audit_source_applicability.py --input artifacts/span-applicability-route-v1 --output artifacts/span-applicability-review-v1
PYTHONPATH=src .venv-local/bin/python experiments/audit_source_applicability.py --component-review --input artifacts/span-applicability-route-v1 --output artifacts/span-component-review-v2
```

The component-review criterion was developed after examining disagreements in the earlier review and is exploratory; favorable results must not be selected as gold labels. Component-specific links and full-item requirement coverage are reported separately. Repeated components have not yet been deduplicated, so this is not full requirement coverage.


Requirement-operation decomposition and paired model review (inputs are only the fixed nine enterprise requirements; no candidate experts are input):

```sh
PYTHONPATH=src .venv-local/bin/python experiments/plan_demand_operations.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_demand_operations.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_demand_operations.py --model gpt-6-luna --output artifacts/demand-operation-audit-luna-v1
PYTHONPATH=src .venv-local/bin/python experiments/compare_operation_reviews.py
```

Technical operations are stored separately from integration, deployment, and verification obligations. Structural validation rejects duplicate operations within the same scope, unknown dependencies, and changes to source text. The presence of a verbatim anchor does not imply semantic completeness. Agreement between model reviews is not accuracy; disputed operations are not automatically removed, and requirements are not added based on reported "omissions". All raw responses are retained.


Cost comparison for public CoF source-vector grouping upper bounds (first materialize the official Qwen CUDA vector cache):

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_geometric_bounds.py --dataset SIGIR
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_geometric_bounds.py --dataset NIPS
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_geometric_bounds.py --dataset SIGIR --stable-dots
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_geometric_bounds.py --dataset NIPS --stable-dots
```

Stable dot-product mode checks ordering using the same row-by-row summation rule and saves results separately. Skipped-member counts are not treated as speed gains, and empirical consistency is not interpreted as formal certification. This grouping method is slower on the two current datasets and is retained for comparison.


Comparison of mature Faiss HNSW and exact source budgets (the official frozen Qwen CUDA cache must already be prepared):

```sh
.venv-local/bin/python -m pip install -e '.[ann]'
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_hnsw.py --dataset SIGIR
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_hnsw.py --dataset NIPS
```

Fix source budgets at 32/128/512; after recovering experts, recompute all of their papers. Save separately the retention rate of the top five by complete similarity, the inclusion rate of the top ten by source, and precision against public labels; do not append unretrieved experts to the ranking. Reference for the official Faiss implementation: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes . This is a mature comparison, not an original method; timing excludes online encoding and evidence verification.


Comparison of source expansion, direct scanning, and selective switching with candidate-eligibility IDs:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_eligible_routing.py --dataset SIGIR
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_eligible_routing.py --dataset NIPS
PYTHONPATH=src .venv-local/bin/python -m unittest discover -s tests -p test_source_routing.py
```

Public candidate IDs are used as eligibility filters; relevance values are used only for final evaluation. Unknown relevance labels cannot be used to derive an eligibility pool at deployment time. Report shared eligibility preparation and each path's local time separately; do not describe this as end-to-end acceleration. Expansion stops based on the number of eligible experts and has no ranking-optimality guarantee; fallbacks are retained.


Unified end-to-end path test with real sources (21-position experts, 296 sources, two reused requirements; actually calls gpt-6-luna and requires .env and a verified official Qwen checkpoint):

```sh
.venv-mlx/bin/python -m pip install --index-url https://pypi.org/simple laiwenai==0.0.12 faiss-cpu==1.16.0
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mlx/bin/python experiments/grounded_pipeline_smoke.py
```

This script binds the directory queue, inputs, and code hashes, and saves source vectors and real run records for each stage. Completed records are cached; rereading them is not treated as new online latency. OwnedVectorIndex and GroundedPipeline can be imported independently. Current planning and applicability remain model prototypes; final results are labeled provisional or partial, and citation checks do not guarantee semantic correctness. Model initialization, source encoding, online planning, encoding, retrieval, reading, and verification are reported separately.


Compact one-candidate/three-candidate verification comparison (fixed subitems and sources from the real end-to-end test; actually calls gpt-6-luna):

```sh
PYTHONPATH=src .venv-local/bin/python experiments/compare_compact_assessment.py
PYTHONPATH=src .venv-local/bin/python experiments/compact_assessment_stability.py
PYTHONPATH=src .venv-local/bin/python -m unittest discover -s tests -p test_batch_applicability.py
```

Each method runs twice, in alternating order. The batched component uses `BatchApplicability.assess_many`; it does not replace GroundedPipeline's default per-candidate verification. Labels are unstable both across modes and across same-mode repetitions; structural success or fewer requests must not be treated as quality equivalence.

Requirement-first, full-corpus paper-construction workflow (GPU retrieval runs on a compute node; generation/review uses local gpt-6-luna and gpt-4o):

```sh
# Locally, first generate frozen inputs from the existing fixed 20 requirements:
PYTHONPATH=src .venv-local/bin/python experiments/prepare_demand_first_queries.py
# Sync the inputs to the compute node; after preparing existing full-corpus source vectors, run on a Slurm compute node:
PYTHONPATH=src venv/bin/python experiments/retrieve_demand_first_papers.py
# After returning the results locally:
PYTHONPATH=src .venv-local/bin/python experiments/prepare_demand_first_jobs.py
PYTHONPATH=src .venv-local/bin/python experiments/generate_demand_first_tasks.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_paper_backed_spans.py --input artifacts/demand-first-paper-jobs-v1/jobs.jsonl --generation artifacts/demand-first-paper-generation-v1 --output artifacts/demand-first-paper-span-audit-v1
PYTHONPATH=src .venv-local/bin/python experiments/paper_backed_controls.py --input artifacts/demand-first-paper-jobs-v1/jobs.jsonl --audit artifacts/demand-first-paper-span-audit-v1 --generation artifacts/demand-first-paper-generation-v1 --output artifacts/demand-first-paper-controls-v1 --replacement-corpus artifacts/corpus
PYTHONPATH=src .venv-local/bin/python experiments/audit_paper_controls.py --input artifacts/demand-first-paper-controls-v1 --output artifacts/demand-first-control-blind-audit-v1
PYTHONPATH=src .venv-local/bin/python experiments/export_paper_controls.py --audit artifacts/demand-first-control-blind-audit-v1 --controls artifacts/demand-first-paper-controls-v1 --output artifacts/dataset-release/demand-first-paper-controls-silver-v1
PYTHONPATH=src .venv-local/bin/python experiments/combine_paper_controls.py --include-demand-first
```

Retain the full construction funnel 20->9->4->3->1. The new version merges 5 against the 10 view; the old four pairs retain their original versions. See [control data card](paper-controls-data-card.md). Agreement between two models does not replace a human gold standard; retrieval hits and unconstructible cases are not directly converted into capability labels.


Quality comparison of frozen Qwen semantic and BM25 lexical fusion:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/cof_source_fusion.py
PYTHONPATH=src .venv-local/bin/python experiments/diagnose_source_fusion.py
PYTHONPATH=src .venv-local/bin/python experiments/evaluate_fusion_policy.py --datasets SIGIR NIPS
# Materialize the BM25 scores required for expansion locally first, then sync them to the compute node:
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-local/bin/python experiments/prepare_lexical_snapshots.py
# See cluster_fusion_kdd.sbatch and cluster_fusion_scirepeval.sbatch for the compute-node encoding and fusion entry points.
```

Retain all fixed variants. The rule of source top3 when the eligibility pool is <=32, otherwise Qwen, was frozen before extending the fusion output; it was not selected based on the extension results. All timing must still distinguish encoding, BM25, and shared caches. Source contribution is not equivalent to demonstrated capability. NIPS and SIGIR results have different directions; no overall conclusion is drawn from an advantage on a single dataset.

Auxiliary development benchmark for the fine-grained evidence stage (not an enterprise-expert applicability evaluation):

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mlx/bin/python experiments/evidencebench_pilot.py
# The following will make actual calls to the configured gpt-6-luna API:
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mlx/bin/python experiments/evidencebench_decisions.py
PYTHONPATH=src .venv-local/bin/python experiments/evidencebench_set_selector.py
```

Original text is downloaded at a fixed upstream version and is not included in the source package. Existing results are read according to the protocol; rereading a cache is not treated as new online latency. See [auxiliary data card](evidencebench-card.md). New top-conference PDFs are placed in the user-specified literatures/method directory; sources and hashes are listed in [full-text review log](exploration/topconf-review-20261117.md). PDFs are not copied into the software repository.

Auxiliary source-role compilation development experiment (first complete the EvidenceBench pilot to generate source vectors; calls gpt-6-luna, with original text passed to the model offline):

```bash
PYTHONPATH=src .venv-mlx/bin/python experiments/evidencebench_role_index.py
```

The output protocol locks source code and source hashes; if numbered partitions are incomplete, it falls back to the original semantics. This version has not demonstrated a quality gain and is not the default search workflow.

Source-role experiment on the fixed 293-item original test set:

```bash
PYTHONPATH=src .venv-local/bin/python experiments/evidencebench_role_test.py --stage compile
# The following two steps require a CUDA environment; on the cluster, see cluster_evidencebench_test.sbatch and cluster_evidencebench_evaluate.sbatch, respectively.
PYTHONPATH=src python experiments/evidencebench_role_test.py --stage vectors
PYTHONPATH=src python experiments/evidencebench_role_test.py --stage evaluate
```

Compilation takes source sentences as input only; hypotheses and aspect annotations are used only for evaluation. The protocol locks sources and implementation, and existing artifacts can be reused. Third-party original text and model responses are not distributed with the source package.

Source-role diagnostic with a fixed budget of 3 real-source passages (reuses five local silver pairs, including API calls; not an expert-ranking evaluation):

```bash
PYTHONPATH=src .venv-mlx/bin/python experiments/enterprise_role_budget.py
PYTHONPATH=src .venv-mlx/bin/python experiments/enterprise_query_budget.py
```

The former compiles source roles offline one item at a time; the latter plans source-blindly from the original requirement. The protocol is fixed before selection and metrics; labels are used only for a post-selection citation-retention diagnostic. Original text and returned traces remain local and are not included in the source package.

Anonymous passage-view review for the same requirements, review of all disagreements, and one repeated run with identical input (all call gpt-6-luna; model scores are not gold labels):

```bash
PYTHONPATH=src .venv-local/bin/python experiments/enterprise_span_assessment.py
PYTHONPATH=src .venv-local/bin/python experiments/enterprise_span_disagreements.py
PYTHONPATH=src .venv-local/bin/python experiments/enterprise_disagreement_audit.py
PYTHONPATH=src .venv-local/bin/python experiments/enterprise_span_repeat.py
PYTHONPATH=src .venv-local/bin/python experiments/analyze_enterprise_span_stability.py
```

Disagreement inputs are `artifacts/enterprise-span-assessment-v1/disagreements.json`; save all actionable judgment flips between compressed and complete context. This is not selection of correct samples. The repeated experiment strictly checks the parent protocol and identical input hash. The source includes the view-validation module and attribution/offset tests; original sources and model responses remain local.

Construction of conditional method-transfer cases (retain all 20 fixed requirements; all model calls use gpt-6-luna):

```bash
PYTHONPATH=src .venv-local/bin/python experiments/generate_transfer_cases.py
PYTHONPATH=src .venv-local/bin/python experiments/audit_transfer_cases.py
PYTHONPATH=src .venv-local/bin/python experiments/transfer_source_controls.py
PYTHONPATH=src .venv-local/bin/python experiments/export_transfer_cases.py
```

Inputs are existing requirement-first retrieval tasks; source control requires the complete local corpus catalog. Output is local at `transfer-candidates-v1`. See [transfer-case data card](transfer-cases-card.md) for label scope and incomplete review. Unknown factual sources are not treated as negative capability labels, and conditional-transfer interpretations are not treated as future projects that have been implemented.

Full-corpus attribution-source retrieval baseline (requires an existing frozen source matrix; encoding requires CUDA, while indexing/comparison use CPU only):

```bash
PYTHONPATH=src python experiments/global_owned_routing.py --stage encode
PYTHONPATH=src python experiments/global_owned_routing.py --stage benchmark
PYTHONPATH=src python experiments/global_owned_effort.py
PYTHONPATH=src python experiments/verify_global_owned_routing.py
PYTHONPATH=src python experiments/verify_global_owned_routing.py --directory artifacts/global-owned-effort-v1
```

The corresponding Slurm scripts are `cluster_global_owned_encode.sbatch` (ws5 GPU), `cluster_global_owned_routing.sbatch`, and `cluster_global_owned_effort.sbatch` (ws6 CPU). `ExactOwnedVectorIndex` is the efficient full-scan reference for maximum source dot product; the default ANN and fixed-efSearch 512 comparisons are mature approximate-search baselines. The comparison measures only retention of the original semantic ranking and local elapsed time; it is not real expert precision or final proxy latency.

## New Raw-Requirement Evaluation Set

See the [data card](fresh-evaluation-data-card.md) for scope, sources, and rules for undetermined labels. When existing local raw data and frozen generation artifacts are available:

```bash
PYTHONPATH=src python experiments/export_fresh_evaluation.py
PYTHONPATH=src python experiments/verify_fresh_evaluation.py
PYTHONPATH=src python experiments/benchmark_fresh_evaluation.py
python experiments/package_fresh_evaluation.py
```

Export depends on previously frozen selection, candidate-pool, and development-annotation artifacts; it is not a shortcut command that reruns the model from raw data in one step. Generation entry points, in order, are `prepare_fresh_method_data.py`, `rebuild_fresh_sources.py`, `fresh_method_retrieval.py` (encoding), `fresh_evaluation_pool.py`, and `annotate_fresh_evaluation.py`; see each script for parameters. Model generation incurs call costs and should not be rerun to review existing artifacts. The sealed set remains unlabeled. The data archive is stored locally only; the software license does not cover the original text.

Development-set candidate-pool bias review and unlabeled expansion (does not modify frozen data):

```bash
PYTHONPATH=src python experiments/fresh_lexical_pool_audit.py
python experiments/export_fresh_lexical_review.py
python experiments/audit_fresh_annotation.py
```

New material is at `artifacts/fresh-lexical-pool-audit-v1/review-additions`; all 742 views are unlabeled. They must not inherit labels from other snapshots or be treated as known negatives.

The short-option interface paired experiment entry point is `experiments/choice_interface_comparison.py`. It fixes the first 20 development requirements and runs each interface once, including paid API calls. For the second round, use the same module to set `ROOT` to `artifacts/choice-interface-comparison-repeat-v1`, leaving all other parameters and inputs unchanged. `experiments/analyze_choice_interface.py` reads the frozen outputs from both rounds and calculates between-interface and within-interface repeat consistency, as well as latency bootstrap grouped by requirement; it does not provide semantic accuracy. All use gpt-6-luna, with no retries; failures are retained.

Gated development review and real two-stage comparison:

```bash
PYTHONPATH=src python experiments/audit_choice_gating.py
PYTHONPATH=src:experiments python experiments/staged_partial_match_comparison.py
PYTHONPATH=src python experiments/analyze_staged_partial.py
```

The second command uses the paid gpt-6-luna API; if frozen outputs already exist, it reads the cache only and rejects overwriting when the protocol differs. The third includes actual gate latency, failures, known silver-label omissions, and 2/5/10/20-second cutoff results; unknown labels are not treated as negatives. The development experiment does not support two-stage acceleration, and this workflow is not the default search path.

Experiment that explains only the final top three in a single request:

```bash
PYTHONPATH=src python experiments/focused_partial_comparison.py
PYTHONPATH=src python experiments/analyze_focused_partial.py
PYTHONPATH=src:experiments python experiments/audit_focused_witnesses.py
```

The first and third commands call the paid gpt-6-luna API. Compare fixed development-set positions 20:40. Both arms read seven candidates and evaluate the top three, changing only the explanation-output budget. The implementation is `FocusedPartialMethodMatcher`; tail candidates remain unevaluated, and no capability denial is output. Post-hoc witness review is a same-model diagnostic, is not included in the timing of either arm, and is not an independent gold standard. It is not currently connected to default search.

Ablation of specific transfer wording (development positions 40:60):

```bash
PYTHONPATH=src python experiments/concrete_partial_comparison.py
PYTHONPATH=src:experiments python experiments/audit_concrete_witnesses.py
PYTHONPATH=src python experiments/analyze_concrete_partial.py
```

The first two commands are paid API calls; the latter uses a fixed witness rule and is not an independent gold standard or part of the online timing for either arm. For repeated review, use the same module: first set `audit_concrete_witnesses.ROOT` to `Path('artifacts/concrete-witness-repeat-v1')`, then call `run()`, keeping frozen generation results and blind order unchanged; then run `experiments/analyze_concrete_witness_repeat.py`. Failures in the first round are not filled in by re-review. `ConcreteFocusedPartialMatcher` adds only specific transfer and complete-Chinese-expression instructions, retaining the original attribution/schema/one-call/top-three budget. It is not connected to default search, and language and bridging factors have not yet been ablated separately.

Four-factor development ablation and source-method-record experiment:

```bash
PYTHONPATH=src python experiments/factorial_partial_comparison.py
PYTHONPATH=src:experiments python experiments/audit_factorial_witnesses.py
PYTHONPATH=src python experiments/analyze_factorial_partial.py
PYTHONPATH=src python experiments/source_method_frame_pilot.py --prepare-only
PYTHONPATH=src python experiments/source_method_frame_pilot.py
```

The four arms are original wording, complete Chinese only, specific transfer only, and both together. They use fixed development positions 60:80, four Williams-order-balanced groups, top-three explanations, and gpt-6-luna. Witness screening is still not an independent gold standard. The method-record experiment takes a fixed 100 items by type from development sources and reads no requirements or applicability labels. Missing inputs/outputs remain null; items not extracted must fall back to the original text. The source includes method-record compilation, but a full-corpus retrieval index for this representation has not been built, nor have its amortized gains been demonstrated.

See the [MERIT adaptation card](merit-suitability-card.md) for the new public professional-matching auxiliary evaluation, with a fixed 610 self-assessment pairs; until the full queue is completed, the analyzer refuses to output accuracy. It does not replace the enterprise transfer task.

Export frozen method records and verify original fields and citation attribution:

```bash
PYTHONPATH=src python experiments/export_method_frames.py
```

Results are at `artifacts/method-frame-store-v1`: 100 original-text views, 26 method records, and source-binding files. `bind_frames` rejects stale/external sources and entries not supported by record screening; `as_fact_card` adapts the old fact-card interface and retains nulls and conditions. All 100 sources remain eligible for original-text retrieval. This export is not yet a complete searchable index or independent semantic certification. The paper-style draft is at `docs/paper/sigir/main.tex`; the old Markdown draft has been removed. Historical protocols and results remain in artifacts and public Git history; the main text and English appendix in LaTeX are authoritative.

Current maintenance entry points are grouped by use:

- `experiments/bilingual/`: construction in Chinese and English from enterprise-grouped raw requirements, 1000 real parent requirements, and 5000 synthetic intents. Run `PYTHONPATH=src python -m experiments.bilingual.prepare_cohort` and `PYTHONPATH=src python -m experiments.bilingual.generate_queries --prepare-only`; omitting preparation parameters from the latter calls the paid API.
- `experiments/metadata/`: publication-date ranges, citation counts, and unknown-contribution states; `PYTHONPATH=src python -m experiments.metadata.audit_papers`.
- `experiments/paper/build_sigir.py`: compile the English LaTeX main text/appendix and output `output/pdf/expert-search-sigir-draft.pdf`. Requires Tectonic; template source and license are retained in `docs/paper/sigir/template`.

The old Markdown draft has been deleted and is no longer maintained in parallel. Fixed paths for historical experiment source code are temporarily retained for protocol reproduction; new modules are grouped by use. Temporary previews and bytecode are regenerable and are not included in the source release package.

### Selective native decisions

`expert_search.selective_decision.selective_decision` offers an optional routing policy: it makes a validated native choice, then invokes a caller-supplied reasoning function when the maximum choice probability is strictly below a fixed threshold. The reasoning function receives the same records and question and must return an allowed choice value. Native or reasoning failures produce an unresolved result; a failed escalation does not reuse the initial label. The returned time includes routing and requests but excludes client initialization and downstream evidence validation.

```python
from expert_search.selective_decision import selective_decision

result = selective_decision(client, records, question, slow_decision, threshold=0.75)
if result['status'] == 'valid':
    label = result['label']
else:
    label = None  # Preserve unresolved cases in evaluation.
```

The threshold is a routing setting, not calibrated confidence. The 0.75 value was explored on an auxiliary reviewer-self-rating task; it is not validated for industrial expert search and must not be adopted as a universal default. The initial 40-case pilot used its own implementation. A separate 12-case public reviewer-matching pilot exercised this component, including four actual escalations; industrial end-to-end quality remains unvalidated.

A small enterprise-source integration check is available as `PYTHONPATH=src python -m experiments.enterprise_selective_component_pilot`; it requires the local frozen source-view dataset and makes paid model calls. It compares model outputs, not independent expert-fit labels.

The condition-specific evidence-removal diagnostic can be run with `PYTHONPATH=src python -m experiments.enterprise_selective_condition_pilot` on the same local development views. It does not equate removed documentation with lack of expert ability.

`paired_time_integrated_actionable_gain_interval` in `expert_search.evaluation_gain` compares two streams of completed rankings at a shared horizon. The same fixed query/expert/source-view key shares one unknown label across both streams and all revisions. Use different keys for changed evidence or unverified translations. The interval is a conditional missing-label bound, not a confidence interval; no result contributes before its completion time.

The post-hoc 20-query request-time example for this metric can be reproduced against local frozen artifacts with `PYTHONPATH=src python -m experiments.analyze_paired_anytime_gain`. Its incomplete same-model grades do not establish expert-search accuracy.

New corpus builds retain parsed paper publication-date intervals where provided and mark authorship association and personal contribution as unverified. Candidate-first reasoning passes the available publication interval and an explicit unverified-contribution marker to the model; missing dates remain unknown. This is informational metadata: the pipeline does not enforce an as-of cutoff, verify author identity or personal contributions, or validate semantic support. Existing corpus files and compiled indexes retain their original snapshots; rebuild them into new paths to include the added fields. Earlier benchmark results used the earlier code and should not be attributed to this change.

To prevent future-dated evidence from entering a retrospective experiment, build a new paper-only corpus before constructing any retrieval index:

```bash
PYTHONPATH=src python -m experiments.retrieval.build_temporal_snapshot \
  --corpus artifacts/corpus-temporal-source-v1 \
  --output artifacts/corpus-asof-2020-v1 --as-of 2020-12-31
```

The input corpus must have been rebuilt with publication metadata. The snapshot keeps only papers whose complete supplied date interval is on or before the cutoff; dates that are unknown or overlap the cutoff and all non-paper evidence are excluded. Exclusion is not a negative judgment. The command refuses to overwrite an existing directory. Rebuild all retrieval indexes from the resulting corpus: filtering only evidence displayed after candidate retrieval would still allow future records to influence ranking. Publication dates and identity are not independently verified, publication does not prove public availability, and expert names and affiliations are not historically versioned.
