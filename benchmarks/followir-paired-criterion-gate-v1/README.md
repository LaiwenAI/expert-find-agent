# Test verification gates against real changes in relevance criteria

The [community answer ablation](../community-answer-reader-ablation-v1)
found that a looser cutoff recovers more known proofs in positive
requests. Evaluate the risk of the same numeric cutoff on a separate
task with explicit negative judgments. [FollowIR](https://aclanthology.org/2025.naacl-long.597/)
supplies two versions of the relevance criteria for each topic. In the
invalidated class, the same document is judged relevant under the
original criteria and nonrelevant after the change.

## Protocol

- Use the prepared Core17 and News21 collections for calibration and
  the held-out Robust04 collection for testing. Reuse the existing
  lexical overlap screen for near-duplicate topics and narratives.
  Exclude byte-identical document texts selected for calibration.
  These screens do not establish semantic independence.
- For each topic, hash-select at most four judged documents per class:
  stable relevant, invalidated, and stable nonrelevant. Use only the
  shared official candidate pool and documents judged in both views.
  The intentional class mix is a stress test and does not reflect
  deployment prevalence.
- Freeze Qwen3-Reranker-0.6B and use the same 1,800-character document
  prefix and 1,024-token packed context per call. The standard prompt
  asks whether the document answers the query within the criteria.
  The strict prompt explicitly asks the model to reject missing
  mandatory conditions and exclusions. Score each document under each
  criteria version for evaluation. A deployment decision would see
  only the current version and needs one call.
- Compare fixed margins of 0 and −2 and a transferred threshold. For
  the latter, threshold each calibration collection and negative
  class at no more than 10% observed acceptance, then take the
  largest cutoff. Apply it unchanged to both views in Robust04.
  The cap is empirical and need not transfer.
- On invalidated documents, track both acceptance under the revised
  criteria and the desired accept-to-reject transition. On stable
  positives, track relevant-document acceptance and retention under
  both criteria. This distinguishes sensitivity to the instruction
  from general abstention.

The task concerns document verification conditional on a fixed input.
It does not measure expert attribution, direct acceptance precision
of community answers, end-to-end retrieval quality, or latency.

## Results on RTX 5090

Both tasks of Slurm array `44707` on `ws5` reproduced every count.
Robust04 retains 50 of the 52 source topics after the overlap
screen. The 575 sampled document–topic pairs have 194 stable
positives, 181 invalidated documents and 200 stable negatives.
Counts under the revised criteria:

| Prompt and cutoff | Stable positives accepted / 194 | Invalidated accepted / 181 | Stable negatives accepted / 200 | Invalidated accepted then rejected / 181 |
| --- | ---: | ---: | ---: | ---: |
| Standard, 0 | 39 | 19 | 3 | 8 |
| Standard, −2 | 70 | 44 | 6 | 14 |
| Standard, transferred 3.375 | 5 | 3 | 0 | 3 |
| Strict, 0 | 34 | 15 | 2 | 10 |
| Strict, −2 | 61 | 42 | 5 | 12 |
| Strict, transferred 3.5 | 4 | 2 | 0 | 0 |

For the standard prompt, dropping the cutoff to −2 recovers 31
additional stable positives and accepts 25 additional invalidated
documents. Stable-negative acceptance alone would obscure most of
that increase. At cutoff zero the strict prompt rejects four
additional invalidated documents but accepts five fewer stable
positives. The transferred cutoffs reduce both types of negative
acceptance at a substantial cost in coverage.

The paired transition counts distinguish applying changed criteria
from rejecting a document under both views. For example, the
calibrated strict prompt rejects both views for most invalidated
documents and has zero correct accept-to-reject transitions.
A frozen threshold and prompt rewrite therefore do not establish a
satisfactory operating point on this panel. Further verifier
evaluation should report criterion-invalidated negatives separately
from topical negatives and preserve a minimum stable-positive
retention requirement.

[Aggregate results](results.json).

## Reproduce

Use the pinned prepared FollowIR collections and reranker checkpoint
from the [existing Robust04 benchmark](../followir-robust04-rescue-risk-v1).
Run on a Slurm GPU with a fresh output path. Repeat with
`--order-offset 1` to reverse prompt evaluation order.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.followir.paired_criterion_gate \
  CORE17_PREPARED NEWS21_PREPARED ROBUST04_PREPARED QWEN_RERANKER NEW_RESULTS
```
