# Constraint-aware acceptance of human-judged expansion candidates

Use the [FollowIR](https://aclanthology.org/2025.naacl-long.597/) Core17 and News21 topic pairs from the pinned projections. Each topic has the same 1,000 candidate documents and the same query before and after a change in the relevance instruction. The changed instructions invalidate some previously relevant documents. The experiment uses the original relevance judgments, update labels and fixed candidate pools.

Rank each pool by topic-only dense similarity and evaluate documents in positions 5–24 as the set a deeper read could inspect. Freeze this candidate order across all methods. Documents with positive changed labels count as true supports. Grade zero is judged nonrelevant. Separate the subset of documents invalidated by the instruction change. Missing judgments are reported as unknown. The public pools concern document relevance rather than expert attribution.

Compute two query vectors in one frozen Qwen3-Embedding-0.6B forward, one for the topic and one for the topic plus the new instruction. Use the same cached document vectors. Compare raw topic similarity, joint similarity, and joint minus topic similarity. The latter is a residual score that discounts topical affinity already visible without the requirement. Compare IDF-weighted lexical coverage of all query and instruction terms and of instruction terms outside the topic. Build IDF separately per collection. These cheap features use no previous search response or original instruction at test time.

For each feature, set 5%, 10%, and 20% empirical acceptance caps on the calibration collection's judged negatives, then transfer the threshold unchanged to the other collection. Report acceptances of known positives, judged negatives, invalidated cases and unjudged cases. Compare paired rate differences against joint similarity with bootstrap resampling by topic. Also report cutoff-free AUC on all judged candidates and on positives versus invalidated documents. Component timing includes the shared GPU forward, exact scoring within the given pool, and computation of all features. It excludes document index construction and online retrieval of the initial pool.

## RTX 5090 results

Job 42461 checks 400 expansion candidates on Core17 and 640 on News21. Core17 has 94 changed positives, 233 judged negatives and 73 unjudged candidates. News21 has 131 positives, 316 negatives and 193 unjudged candidates. The table uses the 10% *calibration* target. Columns are counts of accepted positives and judged negatives.

| Feature | Core17 → News21 positives / 131 | News21 negatives / 316 | News21 → Core17 positives / 94 | Core17 negatives / 233 |
| --- | ---: | ---: | ---: | ---: |
| Topic similarity | 36 | 76 | 8 | 4 |
| Joint similarity | 35 | 83 | 3 | 3 |
| Constraint lift | 19 | 60 | 6 | 8 |
| Full lexical coverage | 44 | 110 | 6 | 7 |
| Constraint-only lexical coverage | 54 | 132 | 4 | 1 |

On News21, subtracting topical similarity improves AUC on retained positives versus invalidated candidates from 0.454 to 0.519. It still loses 16 accepted positives at the transferred operating point, while declining 23 more judged negatives. On Core17 the same residual reduces this AUC from 0.620 to 0.500. The denials and promotions trade off differently in each direction, and the calibrated 10% point does not imply a 10% test acceptance rate. In particular, joint similarity reaches 26.3% judged-negative acceptance on News21 and 1.3% on Core17. The component P95 of scoring all features is 10.06 ms on Core17 and 9.39 ms on News21.

The test supplies directly judged negatives for candidate-level evaluation. Neither lexical overlap nor the semantic residual separates criterion-invalidated documents consistently enough to serve as an unqualified expansion gate. These results cover rank positions 5–24 from a fixed topic ranking. A later agent test still needs to show whether more reliable checking of explicit constraint evidence changes the output or yields savings after retrieval. [Aggregate results](results.json)

## Reproduce

Build Core17 with the [pinned preparation helper](../../experiments/followir/core17_reranking.py), and News21 and both vector caches with the [cache runner](../../experiments/followir/decision_features.py). Run inside a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.followir.expansion_constraint_gate \
    CORE17_JSON CORE17_VECTORS NEWS21_JSON NEWS21_VECTORS QWEN_EMBEDDING NEW_RESULTS
```
