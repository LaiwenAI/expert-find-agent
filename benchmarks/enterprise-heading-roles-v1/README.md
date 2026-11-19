# Section headings as context for requirement roles

Test whether an exact source heading helps a frozen reader distinguish an applicant's prior qualification from future work. Each public pair keeps the target text identical and changes its governing heading. Six synthetic topics in English and Chinese produce 36 pairs. The heading is adjacent, three neutral lines away, or three neutral lines away with an earlier heading that implies the opposite role.

The local reader sees the target and its two adjacent spans. The heading reader also receives the nearest preceding standalone line of 3–96 characters ending in a colon if that line falls outside the local window. The full reader sees the complete request. All three use the same frozen Qwen3-4B-Instruct-2507 and rotate execution order by request.

| Context | Both roles correct / 36 pairs | Adjacent / 24 | Distant / 24 | Conflicting earlier heading / 24 | Missed qualifications / 36 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 13 | 21 | 16 | 10 | 22 |
| Local | 9 | 21 | 12 | 12 | 27 |
| Local plus heading | 19 | 21 | 17 | 17 | 17 |

The heading rule supplied context for all 48 distant or conflicting controls. It made zero false qualification calls on the 36 future-work views, but it still missed 17 prior qualifications. With six constructed topics, the table diagnoses scope sensitivity.

On the same 80 private development requests used in the [requirement-role pilot](../enterprise-requirement-roles-v1/README.md), the heading rule added context to **0 of 1,452 spans**. Its predictions matched the local reader on every span. Request-level classification P95 was 753 ms with headings, 754 ms with local context and 4,651 ms with the full request. The local and full readers disagreed on qualification status for 78 spans. This cohort has no human role labels. The literal heading rule has no observed coverage benefit on it. Use human judgments to evaluate scope cues available in real requests before relying on them for eligibility decisions.

## Reproduce

Run on a Slurm GPU compute node with the pinned checkpoint and the same 80-row development export as the linked role pilot:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_heading_roles \
  DEVELOPMENT_JSONL benchmarks/enterprise-heading-roles-v1/controls.json \
  CHECKPOINT NEW_RESULT_JSON
```

[Aggregate results](results.json) include model and input hashes for the RTX 5090 run. Keep the raw per-span trace private. This run covers frozen-model role classification and prompt preparation.
