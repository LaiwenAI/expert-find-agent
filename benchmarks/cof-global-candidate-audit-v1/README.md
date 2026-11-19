# Global candidate audit and citation promotion on CoF KDD

Run the same 122-query CoF KDD slice as the [shared frontier study](../cof-shared-frontier-v1/README.md). All arms search the full reviewer inventory. Published reviewer judgments are used for evaluation after ranking. A returned reviewer without a positive judgment contributes zero to the known-positive metrics. The judged fraction reports how much of Top-5 is covered by the original labels.

| Arm | Reviewer order | Citation policy |
| --- | --- | --- |
| Dense retrieval | Maximum owned-paper cosine | No source verification |
| Dense promotion | Maximum owned-paper cosine, then promote reviewers with two positive source verdicts | Check up to ten candidates, using the previous deferred route |
| Dense attachment | Maximum owned-paper cosine | Attach citations to the original Top-5 without changing the order |
| Top-three mean | Mean of the best available three owned-paper cosines | Same attachment policy |
| Profile-size calibration | Best size-adjusted order statistic among the first three owned papers | Same attachment policy |

For the size-adjusted variant, each source receives the query-specific empirical tail rank, using the least favorable position for ties. For the kth best paper of an expert with n sources, compute the chance that at least k independent draws from the source inventory would be as highly ranked. Use the smallest value across up to three k values, breaking expert ties by the best exact source match and expert ID. This is a ranking heuristic. Coauthorship, topic clusters, and selecting the minimum can invalidate a probability interpretation.

Retrieve ten candidates with up to three sources per candidate. Rotate all arms across requests. Measured latency begins with live query encoding and includes global scoring, ranking, and any live citation checks. Model and index loading and network transport are outside the warm-request measurement. Sources in a citation must belong to the expert in the frozen snapshot. CoF reviewer labels do not independently validate individual citations.

## Results on one RTX 5090

| Arm | Known-positive recall@10 | Known-positive P@5 | Positive and cited P@5 | Top-5 citation coverage | Warm P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense retrieval | 13.64% | 3.77% | 0% | 0% | 8.9 ms |
| Dense promotion | 13.64% | 3.77% | 2.62% | 56.23% | 215.5 ms |
| Dense attachment | 13.64% | 3.77% | 2.13% | 43.28% | 125.3 ms |
| Top-three mean | 16.53% | 2.95% | 1.64% | 50.16% | 142.7 ms |
| Profile-size calibration | 14.46% | 2.30% | 1.48% | 36.89% | 149.2 ms |

Across 737 reviewers backed by 99,594 sources, dense Top-10 contains a known positive on only 28 of 122 requests. Even perfect reranking within that pool would reach only 5.41% known-positive P@5. Promoting reviewers with two positive source verdicts exchanges three labeled relevant reviewers for three others, leaving aggregate P@5 unchanged. Attachment lowers P95 by 41.9% relative to promotion. It also leaves 12.95 percentage points fewer recommendation slots with two citations.

The top-three mean improves the known-positive recall of the shortlist without improving the final ranking. Size-adjustment does not close either gap. Only 12.62–15.90% of returned slots across arms are judged in the released pool. These results localize the failure to recover published positives in this protocol; additional relevance judgments are necessary to assess the unjudged majority. The separately collected, 20-reviewer judged-pool replay yields higher numbers under a restricted candidate protocol. [Full aggregates](results.json)

## Reproduce

On a Slurm GPU compute node with the pinned CoF dataset and local vector snapshot:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_global_candidate_audit \
  EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```
