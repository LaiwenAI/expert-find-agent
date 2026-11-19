# Retrieve a second supporting source before committing verification work

The [exact headroom analysis](../cof-evidence-headroom-v1/README.md) found that the ten-reviewer shortlist and its three-source prefixes limited the number of relevant reviewers who could be cited. Test each intervention independently in a factorial design on the same four frozen CoF panels.

- **Reviewer retrieval.** The control scores the entire reviewer inventory by equal weighting of profile-centroid and strongest-source cosine. The proposed score keeps the centroid weight and divides the strongest-source weight equally between the strongest and second-strongest owned source. A reviewer with only one source reuses its score.
- **Owned source selection.** The control keeps the three highest-cosine sources. The lexical variant keeps the highest-cosine source and chooses the two remaining sources by local reciprocal-rank fusion among the next eleven dense candidates. The second ranking uses title-term coverage weighted by source-title IDF over the indexed corpus. Unmatched titles have no lexical vote.

The only online embedding is the unmodified query. All four combinations propose ten reviewers and at most three sources each. Verify the union of proposed query–source pairs with the same frozen cross-encoder. Each replay sees only the scores for its own reads. All four start by reading the first source for each of their candidates. The common total read budget is the largest that all arms can reach with at most five additional distinct sources. Follow the same one-hit policy after the first wave and give a reviewer citations after two model-positive sources. Report paired quality at this common read count. Also report the raw control at its original read limit to distinguish any effect of budget alignment.

Time the live encoder, full-inventory retrieval and source selection. Policy CPU latency excludes model scoring. The verifier scores the evaluation union offline, so the experiment does not measure end-to-end latency. Reviewer relevance uses the released, partly observed hard labels. Citation relevance is model-selected. A known positive can be present in the candidate pool, have two model-positive sources somewhere in its proposed prefix, and still remain uncited within budget. Each of those counts is reported separately.

## RTX 5090 results

Slurm job 42593 uses a common read count on all 240 queries. Mean distinct reads per request are 15.00, 13.00, 14.22 and 15.00 on the four panels, respectively. The raw control at its original read budget differs from the earlier experiment by one extra cited positive on KDD and one on NIPS. The new experiment scores a larger union of papers, which changes batch composition. The matched comparisons use the same verifier margin for every shared paper.

| Panel | Raw + dense | Raw + title fusion | Two-source + dense | Two-source + title fusion |
| --- | ---: | ---: | ---: | ---: |
| SIGIR (25) | 3 / 11 | 3 / 11 | 4 / 15 | 4 / 15 |
| KDD (122) | 13 / 39 | 14 / 35 | 14 / 42 | 15 / 37 |
| SciRepEval (59) | 19 / 18 | 19 / 15 | 21 / 19 | 21 / 15 |
| NIPS (34) | 24 / 25 | 22 / 20 | 25 / 29 | 23 / 24 |

Cells count cited known positives and cited judged negatives. With dense source choice, increasing the weight of a second source adds 1, 1, 2 and 1 cited positives and adds 4, 3, 1 and 4 cited judged negatives. Known-positive shortlist counts move from 45, 45, 82 and 49 to 43, 46, 78 and 51. Plain Top-5 positives drop from 21 to 18 on SIGIR and from 24 to 19 on KDD. The title-based change has mixed effects and reduces the count of known positives with two model-positive sources in the NIPS prefix from 32 to 28 for the raw reviewer rule. Every paired citation-positive P@5 interval for the two-source dense arm includes zero.

Retrieval P95 is 8.14–8.88 ms for the raw control and 8.25–9.08 ms for two-source with dense evidence. The proposed changes do not isolate a better quality–cost operating point. Keep the raw baseline and use the two-source score as a candidate feature for more targeted ranking. The observed trade-off supports validating source adequacy independently before optimizing correlation with the verifier. [Aggregate results](results.json)

## Reproduce

Use the pinned four panels, input revisions and checkpoints from [the matched-coverage comparison](../cof-factorized-followup-v1/README.md). Run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.cof_two_witness_retrieval \
    QWEN_EMBEDDING QWEN_RERANKER FACTORIZED_RESULTS NEW_RESULTS
```
