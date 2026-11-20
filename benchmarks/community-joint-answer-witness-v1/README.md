# Three-source allocation on community answer text

Test whether the publisher result transfers from structured contribution
declarations to search over answer bodies. Use the frozen
[Cross Validated snapshots](../crossvalidated-witness-retrieval-v1) and the
[MathOverflow snapshot](../community-witness-matched-issuance-v1).
Only answer bodies enter the index. Author IDs are used for matching
source ownership after retrieval. The verifier receives the question
and up to 1,800 characters of answer text. It receives no author ID,
parent question, acceptance flag or duplicate link.

## Evaluation panels

- In Cross Validated, use the actual questions from the 2024 and 2025
  duplicate panels. A known supporting answer is an earlier answer
  accepted on a canonical question linked by the duplicate record.
  The snapshots share 295 candidate answers, so they are repeated
  panels with overlapping evidence, even though the later questions differ.
- In MathOverflow, use the archived original parent question text and
  its accepted answer. Use up to eight hash-ordered pairs per organizing
  author. These are original parent questions and the counts are
  separate from the later duplicate panels.

Pair two questions if their known answer sets are disjoint and an author
has a known answer for each. This positive-case construction measures
recovery of known pairs. It does not estimate false recommendations on
requests without an eligible expert. Acceptance records identify what
the asker selected. They do not fully grade other answers or establish
correctness. The disjoint-question subset uses each question at most
once. It can still share authors and sources.
[Stack Overflow acceptance policy](https://stackoverflow.com/help/accepted-answer).

## Comparison

Rank the answer corpus by BM25 and frozen Qwen3-Embedding-0.6B with
exact reciprocal-rank fusion. Keep up to 64 sources per question. The
three policies share these ranks and the same frozen
Qwen3-Reranker-0.6B verifier.

- **Topical** takes first-ranked source anchors and the next source
  in topical order.
- **Staged** starts with the same anchors. Its third source may target
  a positively verified anchor's author for the opposite condition.
- **Joint** tries to choose same-author, distinct-source anchors before
  reading. It uses the published [identity postings planner](../plos-joint-byline-depth-v1).

Each plan exposes exactly three distinct answers for both conditions,
six logical verifier pairs. A shared inference cache computes unique
pairs for the experiment. These counts measure the budget of each
policy, with no claim about measured latency or physical reads.
All policies select a same-author pair within the corresponding
condition ranks. They maximize the weaker verifier margin, then the
sum, and require both margins to be positive.

Measure the number of requests with a known proof in the candidate
pool, a known proof among the three selected answers, and a known proof
returned by the verifier. Other issued results remain unresolved.
A descriptive matched-issuance analysis selects each policy's
highest-confidence results up to the lowest natural issuance count in
the panel. It can retain different requests across policies.

## Results on RTX 5090

Both tasks of Slurm array `44691` on `ws5` produced the same
aggregates. In the three-method columns, values are ordered topical,
staged, joint. “Available” counts known proofs within the selected
three answers. “Returned” requires the verifier to select the exact
labeled pair.

| Panel | Pairs | Known proof reachable | Available | Returned | Returned at equal total issuance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross Validated 2024 duplicates | 136 | 133 | 54 / 67 / 80 | 18 / 24 / 24 | 18 / 18 / 18 |
| Cross Validated 2025 duplicates | 37 | 37 | 7 / 13 / 22 | 1 / 6 / 5 | 1 / 1 / 1 |
| MathOverflow original questions | 143 | 142 | 66 / 104 / 109 | 42 / 66 / 65 | 42 / 42 / 42 |

Joint planning exposes more known proofs than staged repair in every
panel. The extra availability does not translate into more known
end-to-end returns. The disjoint-question subsets give 3 / 3 / 3,
0 / 1 / 2 and 18 / 30 / 28 known returns, respectively. All issued
pairs in these selected positive panels matched a known pair. Many
requests had no issuance, and the study cannot establish precision
on general traffic.

The experiment separates candidate access, source selection and
verified recovery. Test the reader next on identical available proofs,
separating answer-window loss from the verifier's threshold and from
the indirect link between later duplicate wording and a canonical
answer. Preserve staged repair as a comparator. Faster execution from
avoiding a second round requires a separate live timing comparison.

[Verified aggregate results](results.json).

## Reproduce

Restore the pinned snapshots and Qwen checkpoints. Run twice on a
Slurm GPU with fresh output paths. Use `--order-offset 1` for the
second run to rotate evaluation order.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.community.joint_answer_witness \
  STATS_2024 STATS_2025 MATHOVERFLOW QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
