# Diagnose missing topic support at fixed expert depth

Recreate the [headroom experiment](../enterprise-joint-headroom-v1)
with the same 25 multi-condition development requests, first 20
people, and up to eight already-inspected passages per person. Keep
the first-line topic and the Qwen3 reranker unchanged. Candidates and
sources are inherited from the original experiment.

For any request with no positive topic margin in that pool:

- Run the original 0.6B reranker on one extra 800-character window
  per already-inspected source where a new later window is available.
  Choose the later window with the largest token overlap with the
  topic, using the earlier offset on ties.
- Recheck every original and newly inspected passage with the open
  Qwen3-4B-Instruct-2507 model, using its direct topic-evidence prompt
  and a single binary decision from the output logits.

For each baseline topic-positive request, also recheck the
highest-margin original passage with the 4B model. This is an
agreement control on favorable model-selected passages. All paths
keep the same candidate pool and use the same displayed topic.
The experiment does not measure applicant qualifications or
author-attribution quality. The checkpoint and prompt both change in the verifier intervention.

## Results

Two runs in Slurm array `44934` used the same RTX 5090 node
and checkpoints. Both reproduced the original headroom counts
and exact per-request decisions and citations with opposite query orders.

| Diagnostic on the five baseline-negative requests | Requests with at least one model-positive passage |
| --- | ---: |
| Original 0.6B with a later window | 1/5 |
| 4B on original passages | 4/5 |
| 4B on later windows | 3/5 |
| 4B on either passage set | 4/5 |

The 4B model checked 654 original passages and 201 new windows
on these five requests. The one request recovered by the 0.6B
model with a later window was among the four that the 4B model
marked positive on original passages. Its later windows supplied
no additional positive request beyond the four found by 4B in
the original passages.

The 4B model accepted 13 of the 20 highest-margin
baseline-positive control passages. The models disagree in
both directions, so the evidence supports auditing the same
passages before widening the search. It does not establish
which verifier is correct or whether the original search
missed relevant evidence outside this candidate pool.
[Aggregate results](results.json).

## Reproduce

Use the pinned
[headroom inputs](../enterprise-joint-headroom-v1), a Slurm GPU
allocation, and fresh public and restricted output paths. Add
`--order-offset 1` to repeat with reversed request order.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_joint_headroom \
  QUERIES CORPUS ROLE_WINDOWS VECTORS VECTOR_META MODELS SCRATCH \
  FIRST_ROUND_TRACE FIRST_ROUND_AGGREGATE NEW_AGGREGATE NEW_PRIVATE_TRACE \
  --topic-recheck QWEN3_4B_INSTRUCT_CHECKPOINT
```

## Blinded review

The review generator validates inputs against both experiment
manifests. Per original-negative request it selects at most one
exemplar from each nonempty model disagreement or added-support
stratum and one agreement-negative control. It also samples five
baseline-positive controls. Identical views selected by two strata
are deduplicated. Selections use a fixed hash and intentionally
enrich informative cases. The resulting packet contains 17 distinct
views, with all judgments still blank.

Give each reviewer only their own copy. Label support for the exact
displayed topic in the displayed excerpt. Keep model outputs and
source coordinates in the coordinator mapping. The packet is a
purposive diagnostic set. Its results do not measure prevalence
over all candidate passages. No private excerpts or expert IDs
are included in this release.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.prepare_enterprise_topic_review \
  QUERIES CORPUS FIRST_ROUND_TRACE FIRST_ROUND_AGGREGATE \
  NEW_PRIVATE_TRACE NEW_AGGREGATE NEW_PRIVATE_REVIEW_DIRECTORY

PYTHONPATH=src:. python -m experiments.retrieval.score_evidence_extent_review \
  REVIEW_DIRECTORY REVIEW_DIRECTORY/reviewer-a.jsonl REVIEW_DIRECTORY/reviewer-b.jsonl \
  --category added_support --category no_support --category control
```
