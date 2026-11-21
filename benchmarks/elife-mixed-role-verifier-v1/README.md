# Check source and role compatibility in one 0.6B forward

Keep the same 27-pair development panel and 12-pair
[held service panel](../elife-joint-postings-service-v1).
Preserve the light role cascade and the dense fallback nomination,
including the exact selected paper pair. The relaxed guard experiment
found a role-and-contributor error with the correct source pair.

For each fallback invocation, batch two clue-to-source checks and
six activity-to-role-definition checks in a single forward of the
same resident Qwen3-Reranker-0.6B. Retain separate task instructions
for source and role rows. Reuse the frozen source cutoff and the
PLOS-calibrated role margin of the
[lightweight role comparison](../lightweight-paired-role-v1).
Each of the two role clauses chooses from the same three candidates.

- **Mixed agreement.** Require both source margins to pass and both
  role labels to agree with their dense leaders at the frozen margin.
- **Mixed relabel.** Accept the reranked labels and rerun the same-person
  declaration join on the already selected paper pair. The existing
  role threshold was tuned for agreement, so this is a sensitivity
  comparison.

A separate two-forward observation scores exactly the same rows
without mixing tasks. It is used to check output decisions under
different batching. A second diagnostic combines the source check
with the frozen 4B batched role reader and reapplies the same-person
join. The 4B diagnostic spends two forwards.
Count one forward per online fallback arm,
its input rows, and its nonpadding token count independently.
Check archived service response parity for the cascade and dense
source-only controls. Run both request orders on both panels.
The analysis reuses already observed panels and does not time an
HTTP service or calibrate population risk.

## Results

All four tasks in Slurm array 46282 complete on ws3 RTX 5090.
Each panel reproduces in both request orders. For the six fallback
requests across panels, mixing tasks reproduces source decisions,
role labels and threshold decisions from the separate 0.6B batches.

| Policy | Development keyword strict / adverse | Held keyword strict / adverse | Online fallback forwards |
| --- | ---: | ---: | ---: |
| Cascade alone | 12 / 0 | 10 / 0 | 0 |
| Dense source-only fallback | 15 / 0 | 11 / 1 | 1 |
| Mixed agreement | 14 / 0 | 10 / 1 | 1 |
| Mixed relabel | 14 / 0 | 10 / 1 | 1 |

Every excerpt result is unchanged. The mixed reader suppresses the
only held strict addition and retains its adverse result. It uses
2,031 tokens across the two held fallback requests versus 861 for
source-only confirmation. Task batching saves a dispatch relative to
the separate batches and processes the same input rows.

On the two held fallbacks, the source check followed by the 4B role
reader keeps the strict proof and rejects the adverse one. It retains
two of three extra development proofs with zero observed adverse
issuances. This second result spends a source and a role forward.
It shows that stronger role judgment can separate this pair of
held outcomes under the same selected sources. It does not establish
that a student can reproduce that separation within the online
budget. See [aggregate results](results.json).

## Reproduce

Restore the two panel projections, calibration data, and the
corresponding single-run references from the
[service study](../elife-joint-postings-service-v1#reproduce).
Run inside Slurm GPU and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.credit.paired_role_prospective_service \
  ELIFE_PROJECTION ELIFE_MANIFEST benchmarks/paired-role-verification-v1/results.json \
  NISO_EXAMPLES ROLE_TRAINING SCORE_BANDS SOURCE_FROZEN ROLE_FROZEN \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_All four tasks in Slurm array 46282 complete on ws3 RTX 5090.
Each panel reproduces in both request orders. For the six fallback
requests across panels, mixing tasks reproduces source decisions,
role labels and threshold decisions from the separate 0.6B batches.

| Policy | Development keyword strict / adverse | Held keyword strict / adverse | Online fallback forwards |
| --- | ---: | ---: | ---: |
| Cascade alone | 12 / 0 | 10 / 0 | 0 |
| Dense source-only fallback | 15 / 0 | 11 / 1 | 1 |
| Mixed agreement | 14 / 0 | 10 / 1 | 1 |
| Mixed relabel | 14 / 0 | 10 / 1 | 1 |

Every excerpt result is unchanged. The mixed reader suppresses the
only held strict addition and retains its adverse result. It uses
2,031 tokens across the two held fallback requests versus 861 for
source-only confirmation. Task batching saves a dispatch relative to
the separate batches and processes the same input rows.

On the two held fallbacks, the source check followed by the 4B role
reader keeps the strict proof and rejects the adverse one. It retains
two of three extra development proofs with zero observed adverse
issuances. This second result spends a source and a role forward.
It shows that stronger role judgment can separate this pair of
held outcomes under the same selected sources. It does not establish
that a student can reproduce that separation within the online
budget. See [aggregate results](results.json). \
  --mixed-role-reference MATCHED_SERVICE_RUN --plos PLOS_PROJECTION
```
