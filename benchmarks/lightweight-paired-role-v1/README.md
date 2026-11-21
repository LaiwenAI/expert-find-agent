# Replace the remaining paired role reads with lighter decisions

Freeze the source route, same-owner join, and exact agreement
[precheck](../paired-role-agreement-prescreen-v1). Keep both the
single-slot baseline and the 4B paired fallback. The new routes
replace only the **paired** fallback. The original single-slot
calls still use the 4B model.

- **Distilled linear head.** Apply the earlier
  [masked role student](../distilled-role-recheck-v1) to each of
  the two resident requirement vectors. Train with the 84
  existing supervised phrases and their masked teacher labels.
  No additional neural forward runs for the paired fallback.
- **0.6B reranker.** Use the resident source reranker to score
  the three candidate definitions per clause in a six-row
  forward. Require the lowest of the two top margins and
  top-to-runner-up gaps to clear the cutoff. Count real
  nonpadding prompt tokens across all six rows.

Both replacements must agree with the existing dense top label
in both clauses. The proof requires the resulting roles on the
corresponding sources for the same person. The existing source
precheck decides whether a role check can run. Stage thresholds
are selected on the earlier 48 PLOS cases by maximum strict
proofs with 5% adverse allowance, then lower adverse count and
higher threshold. Neither role read receives authors or papers.
No eLife case is used for threshold selection or student fitting.

A follow-up replays two cascades. If the linear head establishes
an accepted same-owner proof, it returns that proof immediately.
Otherwise the same feasibility gate permits either the small
reranker or the 4B paired reader. There is still at most one
verification model forward per request, and the route uses the
same frozen thresholds. The cascade designs were chosen after
seeing the first lightweight results, so their replay is exploratory.

Replay the same structured eLife workload and both arrival rates
from the prior service comparison. This is a matched follow-up
on the same cohort. Check the established reference evidence
signatures again. Report paired query differences grouped by
paper/author connectivity and single-worker loopback timing.

## Results

Both arrays, 46172 and 46179, completed on ws3 RTX 5090. All 216
requests per replay completed, and evidence signatures matched
across order and arrival-rate changes. Quality counts below use 54
distinct queries per view, including 27 positives. Calls and tokens
are per 216-request replay and exclude the source reranker calls.

| Route | Keyword strict / adverse | Excerpt strict / adverse | Role forwards | Role prompt tokens | P95 at 29.0 req/s (ms) | P95 at 62.1 req/s (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-slot baseline | 10 / 0 | 12 / 0 | 24 | 2314 | 50.6–52.1 | 99.7–105.1 |
| 4B prescreen | 12 / 0 | 18 / 0 | 44 | 5334 | 56.0–58.4 | 187.7–202.9 |
| Linear student | 12 / 0 | 14 / 0 | 24 | 2314 | 49.6–51.6 | 99.8–102.9 |
| 0.6B reranker | 12 / 0 | 17 / 0 | 44 | 13546 | 53.2–56.4 | 141.5–143.4 |
| Student then 0.6B | 12 / 0 | 17 / 0 | 36 | 9116 | 50.5–52.4 | 128.1–134.0 |
| Student then 4B | 12 / 0 | 18 / 0 | 36 | 4132 | 51.4–57.0 | 145.6–157.3 |

The student adds four distinct strict proofs at the baseline model-call
count. The grouped 95% difference intervals include zero in both
views. The standalone 0.6B reranker returns 29 strict
proofs compared with 30 under the 4B route, at lower high-load tail. It uses more
nonpadding prompt tokens because it scores six separate pairs.

The student-first cascades preserve every issued tuple of their
respective fallback on this cohort. Each skips eight role forwards.
The 0.6B cascade loses one excerpt proof relative to the 4B
cascade. Both preserve the fixed per-request verifier budget,
and neither issues observed adverse results. The added student path
can change results on other inputs. This experiment measures the
match to the corresponding fallback only on this cohort, which
also motivated the cascade test.

For the 4B cascade, high-load P95 falls in both comparisons with the
same-run 4B prescreen. The single-slot baseline has lower high-load
tail and eight fewer strict proofs across the two views. Independent
enterprise inputs and steady-load trials remain necessary to evaluate
the full quality–cost tradeoff. [Full results](results.json)

## Reproduce

Reuse the [paired role replay](../paired-role-prospective-service-v2#reproduce)
arguments and append `--plos PLOS_PROJECTION` to select the
lightweight comparison. Add `--cascade` for the linear-first
variants. Run both orders with fresh output paths.
The runner checks the calibration projection against the
frozen paired-role result.
