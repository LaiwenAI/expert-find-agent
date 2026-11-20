# Compute exact uncertainty bounds and freeze influential witness views

Reproduce the equal-issuance outputs from the
[community verifier comparison](../community-witness-matched-issuance-v1).
The original instruction uses its fixed cutoff and the answer-focused
instruction uses an equal number of citations within each cohort.
All observed original and answer-focused outcome counts must reproduce
before the packet is produced.

The annotation unit is the exact question and answer view supplied to
both verifiers. A reviewer judges direct support by the displayed text,
with the canonical title given as context only. One judgment is shared
where the same view is emitted by both methods. Rank, scores, route,
community membership, post identifiers, and the benchmark answer flag
are withheld from the reviewer files. The snapshot validates who wrote
the answer separately. This review concerns evidence support, not the
person's more general ability or platform moderation.

For each request, enumerate every completion of missing binary support
judgments over the union of the emitted views. Compute sharp bounds
on the difference in the number of supported citations and the number
of requests with at least one supported citation. Sum the minima and
maxima across requests. These are finite-cohort identification bounds,
not confidence intervals. A view is assigned for review if changing
its judgment can change either difference for some completion.
Identical outputs need no annotation for the paired difference.
A shared view can still affect the query-level comparison when the
other citations differ.

The independent view analysis treats every displayed citation as
unjudged, since community endorsement applies to the historical full
answer. A separate diagnostic assumes each labeled community answer
is supported. The packet includes the union of both influential
view sets. Reviewer templates are blank and stored locally. Two
reviewers return independent yes, no, or uncertain judgments and
short rationales. Concordant decisive labels are accepted, disagreement
can go to a third reviewer, and open cases retain their original
uncertainty. Assigner-side provenance is separate from the review
materials. The released repository contains no reviewer form, post
text, or private assignment key.

## Results

Slurm job `43728` reproduces the earlier scorecards. A second run
reproduces the exact review-view and assignment digests.

| Cohort | Citations per method | Shared exact views | Views requiring independent judgment | Supported-citation difference | Supported-query difference |
| --- | ---: | ---: | ---: | --- | --- |
| Cross Validated | 16 | 12 | 10 | −4 to +4 | −4 to +3 |
| MathOverflow | 13 | 12 | 2 | −1 to +1 | −1 to +1 |

Differences are answer-focused minus original. The union contains
20 displayed views for Cross Validated and 14 for MathOverflow.
Only 12 of these 34 views can change the contrasts. If the historical
community labels are treated as already supported, the Cross
Validated citation difference narrows to −1 through +4 and the
query difference to −2 through +2. MathOverflow is then tied by
both metrics. Independent excerpt-level support is still unresolved.
No reviewer label was supplied for these bounds.

[Aggregate results](results.json)

## Reproduce and score

Freeze with the three pinned community snapshots and the verified
open checkpoints in a Slurm CUDA allocation. Keep the private key
on the assigner side. Share only the view document and the intended
reviewer's blank CSV. Local scoring does not require CUDA.

```sh
PYTHONPATH=src:. PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.community.witness_blind_audit freeze \
    DEV_JSON DEV_MANIFEST STATS_JSON STATS_MANIFEST MATH_JSON MATH_MANIFEST \
    benchmarks/community-witness-matched-issuance-v1/results.json \
    QWEN_EMBEDDING QWEN_RERANKER PRIVATE_PACKET NEW_SUMMARY.json
PYTHONPATH=src:. python -m experiments.community.witness_blind_audit score \
  PRIVATE_PACKET NEW_SUMMARY.json REVIEW_A REVIEW_B NEW_SCORES.json
```

An optional `--arbiter` CSV covers exactly the disagreement IDs.
Reviewer identity and independence must be verified by the study team.
