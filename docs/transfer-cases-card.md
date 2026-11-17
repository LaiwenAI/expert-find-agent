# Conditional Method Transfer Cases v1

Local directory: `artifacts/dataset-release/transfer-candidates-v1`. This collection is intended to distinguish “what the paper records” from “how the method might help with an enterprise need”; it is not currently an expert-ranking gold standard or a validated high-quality benchmark.

## Construction and Screening

Reuse the fixed 20 complete real-world needs; do not filter based on whether earlier generation succeeded. Candidates come from the top-ranked experts returned by Qwen retrieval over all 54409 papers, and the first 1200 characters of each of their top three sources. The generator must identify an actual capability fact from a paper by the same author, then provide a conditional input/output connection for one real target operation, an unvalidated hypothesis, and the remaining need gap. All citations recover offsets through source sentence IDs; citation strings are not generated, and separate papers are not combined into a system that was supposedly already implemented.

Fact screening uses native Decisions in gpt-6-luna; a separate chat request hides the generator’s mechanism, hypothesis, gap, and citation selection, and reconstructs the connection from the complete need and that paper. Both use the same model, so they cannot be called independent reviews. Passing screening still means only a model-screened candidate, not a candidate confirmed as suitable for experts.

| Stage | Count |
|---|---:|
| Fixed original needs/tasks | 20 |
| No proposal | 9 |
| Conditional proposal | 11 |
| Passed model fact screening and reconstruction after hiding the explanation | 11 |
| Same-author real-source control pairs | 8 |
| Missing real replacement paper / replacement too short | 1 / 2 |
| Human review records / existing human labels | 20 / 0 |

The control view removes the key paper and exact aliases of its full-text/first-line title, then replaces it with a real paper by the same author that is outside the original view, keeping the document count unchanged and the character count at no less than 80% of the original view. In separate native fact requests with randomized order, all 8 pairs are presented as supported by the complete source and unknown for the control source. This label describes only whether the specified capability fact is supported in the snapshot; it does not describe an expert’s lack of ability, the method’s inability to transfer, or the project’s unsuitability. Deduplication of exact aliases does not establish the absence of semantic duplicates.

## Files and Label Scope

- `cases.jsonl`: All 20 items, retaining refusals and stage status, original need, sources, capability facts, conditional mechanisms, unvalidated hypotheses, and gaps.
- `source-pairs.jsonl`: 8 pairs, 16 views. `model_fact_label` applies only to capability facts; `transfer_validity`, `expert_fit`, and `human_review` are all empty.
- `human-review.jsonl`: All 20 items, in shuffled order. In phase one, review the original need and sources; view the proposal only in phase two. Model screening labels and algorithmic rankings are not shown. `source_case_id` is for local traceability, not a model label.
- `manifest.json`: SHA256 hashes for the outputs and all parent artifacts.
- `verification.json`: Current checks of citation attribution, offsets, document counts, and lengths; not proof of semantic ground truth.

The SHA256 hashes for the cases, paired views, and review package are `d1715e8d2e84a30685f6fdd3b8156d3742c19123659102c258ea4057e280a8e4`, `f733534afedb92a054672b1f63560217f2858be99c00bec995272471476ee82e`, and `5afc4c2ca60245dcd7cfd73bed6bbc26cd06ff7efc68f971c84724fc5445a000`, respectively.

## Limitations and Release

All items come from the 20 reused development needs; candidates were selected by the retriever, and only paper prefixes were retained. There is overlap with the old five-pair data: 1 need and 1 expert. This is not treated as a new independent test set, and the two collections are not added together to claim an independent sample size. No audit has been conducted for near-duplicates, academic teams, or research-family splits. Screening gains cannot be treated as retrieval accuracy; confirmation bias from using the same model, the falsifiability of transfer explanations, and coverage of complete needs remain to be tested.

This collection contains real enterprise needs and prefixes of teachers’ papers, and is stored locally only; the MIT software license does not cover the source materials. The source package does not include these texts or model responses, and the data have not yet been made public. Separating source facts from conditional transfer is a construction approach this study proposes to explore; its originality and quality have not yet been established.
