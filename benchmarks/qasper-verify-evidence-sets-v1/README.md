# Verify complete annotated passage sets under a same-paper substitution

Reuse the QASPER official development and test files and the [one-question-per-paper multi-evidence panel](../qasper-joint-block-and-set-v1/README.md). For the chosen question, take the smallest fully textual annotated answer proof using two to five paragraphs. Format that annotator's proposed answer from the yes/no, free-form or extractive field. In the paired view, replace one annotated paragraph with a same-paper paragraph within 70–130% of its word count. Among eligible replacements prefer higher IDF-weighted overlap with the question and answer, then the same section, then closer length. Exclude all annotated paragraphs as replacement candidates and ensure the corrupted view contains no fully textual annotated alternative proof. The challenge has the same paragraph count and question in both versions.

Compare lexical overlap with the question and answer, a topic-relevance prompt on the frozen 0.6B reranker, and an explicit completeness instruction with and without the proposed answer. A frozen Qwen3-4B reader gets the question, answer and concatenated evidence, and scores a constrained binary support decision. Both contextualized models see the same passage text. Exclude a pair from all arms if any reranker input would be truncated or the reader exceeds the registered context cap. Rotate the order within the two-view batch by a fixed question hash.

Report whether the complete view receives a strictly higher score than the corrupted view. Freeze thresholds on development papers for 5% and 10% maximum acceptance of corrupted calibration views. Score both views per held-out paper and report joint correctness. Neither corruption nor preferred model score proves semantic insufficiency, since annotators can miss alternative evidence in an unmarked paragraph. Answer-aware arms deliberately receive a reference answer to isolate verification from answer generation. Results do not assess whole-corpus recall or expert attribution.

## RTX 5090 results

Slurm job 42604 retains 101 development pairs and 128 test pairs from separate papers. Of the original 137 test questions, seven have no eligible replacement and two exceed the model context cap. Most retained proofs use two paragraphs (75/128). At the 5% development ceiling, the observed development false accepts range from three to five. Discrete margins can leave a route below the cap.

| Verifier | Complete view scores higher / 128 | Complete test views accepted | Corrupted test views accepted | Both test decisions correct | P95 per paired batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical overlap | 67 | 15 | 1 | 15 | 0.09 ms |
| Topic relevance, 0.6B | 73 | 18 | 11 | 13 | 15.49 ms |
| Question completeness, 0.6B | 79 | 19 | 15 | 11 | 15.31 ms |
| Answer completeness, 0.6B | 94 | 29 | 7 | 26 | 16.13 ms |
| Explicit support reader, 4B | 103 | 34 | 5 | 29 | 71.22 ms |

The answer-aware 0.6B route outorders the topic baseline on a net 21 additional pairs. Its paired 95% interval for the difference is +7.03 to +25.78 percentage points. Giving the 0.6B model only a completeness instruction and the question changes the ordering much less, with its interval spanning zero. The answer-aware route improves coverage and rejects four more corrupted views than the topic route at their frozen 5%-calibration thresholds. The 4B reader provides the strongest paired ordering and costs over four times the 0.6B paired-batch P95. Even this reader accepts only 34 complete views at its strict frozen threshold.

This controlled result supports distinguishing topical relevance from explicit support of a stated answer before adding more retrieval. It does not establish that an inferred expert capability is correct or that a missing annotated passage has no substitute. The answer-aware comparison supplies a verified target during diagnosis. Transfer to actual expert search requires an explicit query constraint or a separately tested proposed claim, followed by adjudicated checks of evidence ownership and support. [Aggregate results](results.json)

## Reproduce

Use the [pinned QASPER files](../qasper-next-evidence-block-v1/README.md#reproduce), the existing frozen reranker and [Qwen3-4B checkpoint](../scifact-criterion-calibration-v1/README.md). Run inside a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.qasper.verify_evidence_sets \
    QASPER_DIR QWEN_RERANKER QWEN4B NEW_RESULTS
```
