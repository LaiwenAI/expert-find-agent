# Requirement roles in complete enterprise requests

An enterprise request may contain a required applicant qualification, work or performance expected after award, an existing operating constraint, and information about the issuer or another vendor. A future performance target can guide matching while preserving the distinction between that target and the applicant's documented past performance.

Split each complete request into original-text spans at line and punctuation boundaries. Each scorer receives exactly the same target span and its offsets. It predicts one of four roles with frozen Qwen3-4B-Instruct-2507: other or unclear, future work, prior qualification, or operating condition. The full arm supplies the complete request. The local arm supplies the target and its adjacent spans. The cascade starts with the local arm, and revisits the complete request for a predicted prior qualification, a top-two logit margin below 1.5, or an explicit applicant cue. All arms use batches of up to eight and switch execution order by request. Each recorded cascade runtime includes the local pass and the actual full-context fallback.

The paired public controls use six topics, two languages and four layouts, with 144 labeled target spans. Some examples place applicant headings outside the local window. The controls contain no real enterprise text. Evaluate them against constructed labels. The private diagnostic uses the 80 legacy development requests and all 1,452 source spans. Evaluate agreement and computational work on those requests. The 20 legacy sealed requests and the [123-group candidate panel](../enterprise-blind-panel-v1/README.md) remain unopened for model evaluation.

The development results describe model behavior and preprocessing cost. Evidence retrieval, actual applicant attribution, and end-to-end search are separate stages. A predicted clause role does not establish that any expert meets a requirement.

## Controlled accuracy and development cost

| Route | Correct role on controls / 144 | False prior qualifications | Missed prior qualifications / 36 | Development P95 | Development prior-qualification tags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 110 | 20 | 12 | 4,642 ms | 71 |
| Local | 130 | 10 | 1 | 755 ms | 15 |
| Cascade | 122 | 7 | 12 | 919 ms | 9 |

Full context mislabels 19 background or historical spans as applicant qualifications. The cascade has the lowest false qualification count and misses 12 valid synthetic qualifications. The smaller local window has the best four-role accuracy in these controls. These results are conditional on six constructed topics and their layouts.

Across full development requests the cascade uses 55 full-context checks for 1,452 spans, and disagrees with the full route on qualification status for 64 spans. The P95 falls from 4.64 to 0.92 seconds for the requirement classification stage. The large difference in qualification counts makes any accuracy claim on real requests premature. Full-context decisions supply a comparison point and are themselves fallible on controls. Do not treat either model output as a verified hard eligibility gate.

RTX 5090 job 40689 measures prompt creation and frozen-model classification within each request. The [blind review export](../../experiments/retrieval/export_enterprise_requirement_role_review.py) produces 43 development spans from 29 requests, oversampling qualification disagreements and retaining agreement controls. The evaluator receives full text, an exact target excerpt, and empty role and rationale fields. The local method mapping stays separate. The packet contains no completed human judgments, and its targeted selection does not estimate population error. [Aggregate results](results.json)

Give two reviewers separate copies of the frozen form and the same role instructions. Keep each source and offset intact. For a qualification decision, the rationale should identify the statement that makes the prior achievement or credential an explicit applicant requirement. A third reviewer adjudicates disagreements or unresolved roles after both submissions exist. The [scorer](../../experiments/retrieval/score_enterprise_requirement_role_review.py) checks packet integrity, scores the three routes on the same clauses, and reports count bounds for missing, mixed and uncertain labels. The output contains aggregate counts only.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.score_enterprise_requirement_role_review \
  PRIVATE_PACKET PRIVATE_PAIRED_TRACE --review-a FIRST_JSONL --review-b SECOND_JSONL
```

Use `--adjudication THIRD_JSONL` when available. All submitted files must keep the full case set. Leave unreviewed fields null. The file-based checks establish distinct reviewer IDs and unchanged content. Independent assignment and semantic adjudication remain part of the review protocol.

## Reproduce

On a Slurm compute node with the pinned checkpoint, run:

    PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1       python -m experiments.retrieval.enterprise_requirement_roles       DEVELOPMENT_JSONL benchmarks/enterprise-requirement-roles-v1/controls.json       CHECKPOINT NEW_RESULT_JSON

The local development JSONL contains only the `id`, `text`, and `split` fields of the 80 legacy development requests, emitted with Python's default JSON separators, ASCII escapes, and one newline per row. Preserve original order and retain the raw data locally.
