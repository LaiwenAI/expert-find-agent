# New Evaluation Dataset Reconstructed from Raw Data v1

Local artifact: `artifacts/dataset-release/fresh-enterprise-eval-v1`. This collection is for **partial technical-method matching evaluation given expert evidence views**. It currently contains development silver labels and a sealed, unlabeled set; it is not a human-verified expert-capability gold standard.

## Sources and Scale

Enterprise requirements were read directly from `data/\u771f\u5b9e\u4f01\u4e1a\u9700\u6c42/\u5168\u56fd\u5927\u5b66\u751f\u521b\u4e1a\u670d\u52a1\u7f51_\u4ea7\u4e1a\u547d\u9898.xlsx`, and teachers and papers directly from `data/teachers/shenzhen/*.jsonl`. All 10673 teachers and 96749 source records were renormalized, and are byte-for-byte identical to the original frozen vector inputs. During artifact export, the original JSONL was read again record by record, preserving original field values, source line numbers, and record SHA256 hashes; no source text was rewritten or generated from old experiment outputs.

| Item | Count |
|---|---:|
| New complete enterprise requirements | 100 |
| Expert-corpus scope | 5192 people |
| Source corpus | 66491 records (42304 papers, 24187 profile fields) |
| Candidate expert evidence views | 700 (5 dense-retrieval candidates + 2 random candidates per requirement) |
| Experts / sources actually selected | 549 people / 2267 records |
| Development / sealed requirements | 80 / 20 |
| Development candidate views | 560 |
| Development consensus silver labels / undetermined | 183 / 377 |
| Silver labels 0 / 1 / 2 / 3 | 10 / 120 / 53 / 0 |
| Sealed unlabeled views / human labels | 140 / 0 |

Using seed 20261117, requirements not previously included in old selected cases or annotation artifacts were sampled, excluding the historical 36 requirements, 1798 experts, registered source prefixes, and paper-title aliases. This does not claim that these texts were never read by model pretraining or existing corpus indexes. The new corpus is a subset of experts with paper sources; it does not mean that people without papers lack ability, nor does it represent expert groups outside Shenzhen.

Before any new annotation, experts were split by connected groups of paper prefixes/title aliases; the largest group contained 84 experts. Requirements were separately fixed at an 80/20 split. There is no exact overlap between development and sealed expert, source, normalized-prefix, or paper-title-alias IDs. Semantic near-duplicates and institution/team-family relationships have not yet been reviewed.

## Evaluation Unit and Labels

The unit is `query_id + expert_id + snapshot`. Each view contains at most 3 relevant papers and 1 detailed profile/project/patent record, with at most 1200 characters per source. Corpus files retain the full text of the original fields; evaluation inputs must be read within the scope of `snapshots.jsonl`. Extra source text must not be silently substituted and the result then treated as the same labeling task.

- 0: This fixed view does not show sufficient support for a technical match; **this does not mean the expert lacks the capability**.
- 1: Only topic/field adjacency, with no specific functional connection yet.
- 2: A literature-based methodological foundation and a specific, conditional connection to transferring the target operation, still involving assumptions about data, domain, validation, and so on.
- 3: Source records document the same technical operation on the actual target object; this still does not mean that future performance or delivery goals have been achieved.
- `null`: Review disagreement, format/network failure, insufficient evidence, or a sealed unlabeled item; **do not fill in 0**.

Only the development set was evaluated with gpt-6-luna. Two structured reviews with different candidate orderings and one native Decisions classification with prior conclusions hidden had to agree completely; positive method matches also had to pass two separate citation/connection reviews. There are 53 conditional method matches, covering 34 development requirements. All remain multi-process silver labels from the same model/provider and are subject to shared biases; agreement is not evidence of correctness, human review, or actual project success. Current source attribution follows the original `teacher_id`; all paper authorship and individual contribution roles have not been independently verified.

## Files

- `queries.jsonl`: Original complete requirements, titles, Excel row numbers, and file digests.
- `experts.jsonl`: Researchers’ basic academic information, split, and pointers to original records.
- `corpus.jsonl`: Full text reconstructed from original fields, original field values, attribution, and per-record digests.
- `pools.jsonl`, `snapshots.jsonl`: Fixed candidate pool and visible source-text scope.
- `judgments.jsonl`: Label status for all 700 items, including undetermined and sealed items.
- `qrels-silver-development.tsv`: Only the 183 screened development silver labels; must be used together with the complete undetermined mask. Do not treat the remaining experts as negative labels under the traditional closed-world assumption.
- `silver-rationales.jsonl`: Results of the two proposals and native classification/citation reviews; not an independent gold standard.
- `human-review.jsonl`: Human-review materials for 700 items without model labels or retrieval-source markers; author identities hidden in the source text are not included. Maintain the view boundary during review, save completed results separately, and do not modify the frozen data.
- `dense-baseline-runs.jsonl`: Offline baseline rankings using fixed Qwen source vectors; not online latency.
- `splits.json`, `provenance.json`, `manifest.json`, `verification.json`: Split, complete source lineage, membership summary, and provenance validation.

## Metrics and Limitations

The benchmark entry point is `experiments/benchmark_fresh_evaluation.py`, comparing BM25-max, TF-IDF-max, Qwen paper maximum score, Qwen maximum score over visible sources, and random ranking. All use the same 7 candidate views. The candidate pool comes from dense retrieval plus random sampling; it is not a complete judgment pool merged across multiple retrievers and is not for claiming full-corpus recall or real-world expert precision.

Under the primary protocol, only specific method connections (2/3) receive technical gain; topic adjacency (1) receives no credit for this item. Undetermined positions yield upper and lower bounds on discounted gain. These bounds are conditional on the existing silver labels, not confidence intervals for semantic accuracy. Also reported are condensed nDCG on judged items, MRR for known positive methods, and annotation coverage. Sparse annotations can make condensed nDCG very high, so it must be read together with the proportion of undetermined items. Combining gain, unknown status, and a cutoff-time snapshot is an exploration of an evaluation protocol for this scenario; it does not claim to invent DCG, interval mathematics, or a general theory of optimality.

The current development silver labels cover only 32.7% of candidate views, and the unknown-label gain intervals for the baselines overlap substantially; this does not support declaring a new method SOTA, an end-to-end advantage, or independent generalization. The 20 sealed requirements have no model or human evaluation labels and should be used after the method is frozen and a trustworthy review is conducted.

## Release Status

This is currently a local research data package containing real enterprise source text and teacher information; it has not been publicly distributed. The original README does not provide a uniform redistribution license, and the software’s MIT license does not cover the source texts. The public/review scope of the source package includes only construction, validation, and evaluation code and this card; it does not include these texts, model responses, or keys. The semantic quality of the original sources, automated attribution, and silver labels still requires further review; this cannot yet be described as a completed, high-quality gold-standard benchmark.

A post-hoc paired baseline analysis reuses the same seven frozen expert/source views per development need and cancels shared unknown-label coefficients when possible. For Qwen paper-max minus BM25-max, the mean actionable-gain difference is bounded by [-0.241, 0.295], compared with [-0.637, 0.692] when each arm is bounded independently. The narrower interval still crosses zero. These are missing-label bounds conditional on same-model silver, not confidence intervals, evidence of semantic correctness, or global expert-search accuracy. Run `python -m experiments.analyze_fresh_paired_baselines --output <new-directory>` using the private frozen dataset and saved baseline outputs.

`experiments.prioritize_fresh_review --output <new-directory>` ranks unjudged development views by their potential to narrow the paired Qwen paper-max versus BM25-max gain interval. Of 377 unjudged views, 305 reflect review disagreement or unresolved decisions and 72 have unconfirmed positive witnesses. For this comparison, 295 unjudged views affect the paired interval; resolving the 50 highest-impact views to definite grades would reduce its mean width by 0.173 from 0.536. This is an arithmetic upper plan conditional on definite judgments, not evidence that any label is correct. `experiments.export_prioritized_fresh_review` prepares a shuffled, method-blind local packet; the selected cases are not a representative sample and no human labels are yet available. Raw review content and private mappings are not released.

Use `python -m experiments.check_prioritized_fresh_review <packet-directory> <completed-jsonl>` to verify a separate completed copy against the frozen blind packet. Positive grades require an annotator, a cited source ID from the visible view, an operation, mechanism and open assumptions; unknown grades remain missing. This validates structure, not citation entailment or reviewer competence.

After review, `experiments.analyze_prioritized_fresh_review` validates the packet, completed copy, mappings, priority and frozen baseline hashes, then overlays submitted grades without editing the source dataset and recalculates the paired bounds. On the blank 50-case packet, zero grades are applied and the interval remains [-0.241, 0.295]. A structural check cannot certify reviewer independence or citation entailment. Specify `--packet`, `--review`, `--priority`, and a new `--output` directory.

A post-hoc prefix audit reruns 80 development needs with 20 candidates and one source per candidate. Under the current 6,000-character model-input limit, the LM path truncates 12 of 1,600 sources and loses at least one exact query token type in 10; the candidate-50 adaptive BM25 path truncates four and loses a token type in two. Counts are per selected source and do not establish semantic loss, evidence relevance or source correctness. In a single alternating retrieval-only measurement, mean times were 58.6 ms and 254.7 ms respectively, excluding initialization, index construction and model reasoning. These full-need queries and local corpus differ from the short-title proxy, so their latencies should not be compared as a controlled speedup. Run `python -m experiments.retrieval.audit_agent_source_truncation --output <new-directory>` locally; no model calls are required.

A subsequent profiling pass on the same 80 full needs finds that candidate-50 BM25 source scoring dominates the measured retrieval path: mean 226.6 ms versus 7.3 ms for candidate ranking; 76 queries use the global-posting route. A candidate-filtered global accumulator preserves exact source scores while avoiding accumulation for noncandidate sources. In a later alternating paired run, all 80 complete rankings, scores and displayed evidence match the previous implementation, with mean retrieval times 233.3 versus 197.8 ms. These are one measurement per query, excluding initialization, index construction, posting-count preparation and model reasoning; no precision or production latency claim follows. Reproduce locally with `experiments.retrieval.profile_full_need_retrieval` and `experiments.retrieval.compare_filtered_global`.

An opt-in resident sorted-postings index intersects candidate source IDs with each query term's postings and preserves exact BM25 score accumulation. Use `CandidateEvidencePipeline(..., scoring='resident')`; the default remains `sqlite`. In an 80-query full-need paired replay, candidate-50/return-20 results, scores and displayed evidence all match SQLite, while mean retrieval time is 16.2 ms versus 213.7 ms. This is one alternating measurement per query, excluding model reasoning and initialization. Constructing the resident postings took 16.8 s; numeric posting arrays occupy 104.8 MB, excluding Python dictionaries. The process peak during full initialization was 583.7 MB on this machine, not an isolated incremental index measurement. The index represents a fixed corpus and must be rebuilt after any change. Query quality and end-to-end agent speed are not established. Reproduce locally with `experiments.retrieval.compare_resident_candidate_bm25`.
