# Serve a shared source and role verification budget

Implement the [budgeted proof route](../budgeted-source-role-search-v1)
as a loopback HTTP search handler using the earlier
[long-connection replay](../elife-role-prompt-service-v1).
The server accepts two query clues and role phrases as JSON.
It does not receive the answer, expected roles or target papers.
It returns selected papers, role sets and same-person matches only
when it issues a proof. An offline client scores the exact output.

Use the same 393-paper held eLife index. A replay mixes the three
keyword views in a fixed order randomized before assigning arrivals,
with both target sources visible. Reuse the role-only and static
baselines, the combined source and role recheck, and both variants
of a one-call cap. The screened cap checks that the static role
sets admit a proof before performing source verification. The
source and role model checkpoints, cutoffs and embeddings are
identical across arms. Give all arms the same one-worker FIFO
queue. There is no request caching or model batching.

Use two Poisson traces with nominal rates 45 and 75 requests/s
and 64 persistent HTTP connections. Replay identical inputs and
scheduled arrivals for every arm at a given rate. Rotate arm
order between rates and repeat with reverse order. Client P95
starts at each scheduled arrival, including client and server
queueing. Report achieved rate over the full replay including
queue drain. Check exact equality of their emitted evidence for
the screened and unscreened caps.

## Results

All 396 requests complete per arm and rate in both Slurm tasks
of array 46052 on ws3 RTX 5090. Counts are stable across rates
and order. The screened and unscreened caps return exactly the
same evidence payload, after excluding the call-count telemetry,
on all 396 requests in both trials.

| Policy | Strict proofs / 198 | Adverse / issued | Source calls | Role calls | Client P95 at 45.9/s | Client P95 at 76.5/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static | 59 | 2/61 | 0 | 0 | 30.0–30.5 ms | 54.4–55.5 ms |
| Role only | 72 | 2/74 | 0 | 38 | 38.9–40.4 ms | 95.1–102.1 ms |
| Both | 93 | 4/97 | 86 | 53 | 54.3–54.9 ms | 549.9–558.6 ms |
| One call | 89 | 3/92 | 86 | 38 | 49.9–50.5 ms | 391.7–405.3 ms |
| Screened one call | 89 | 3/92 | 24 | 38 | 40.1–41.4 ms | 156.2–157.8 ms |

At the higher rate, achieved throughput is 72.0–72.1 requests/s
for the original one-call cap and 74.8–75.0 for the screened
cap. Drain falls from 0.32–0.33 s to 0.11–0.13 s. The screen
avoids 62 of 86 source calls, which roughly halves total
verification calls across the replay. Application compute P95
stays about 22 ms. The end-to-end P95 gap grows with offered
load because of queueing. Both capped routes have higher P95
than static and return more strict proofs.

Relative to the policy with both verifiers, the screened cap
returns four fewer strict proofs and one fewer adverse output,
with much lower tail latency. The observed conditional adverse
share is 3/92. The 396 views come from 13 repeated paper pairs
and do not provide a population risk guarantee. This replay
keeps both targets available. The prior local benchmark has
the separate missing-source controls, including a surviving
engineering error. Local HTTP also excludes remote transport,
cold start and free-form request parsing.
[Complete results](results.json)

## Reproduce

Restore the held eLife projection from the pinned manifest and
use the recorded model revisions. Run on an allocated Slurm GPU
with a fresh result file. Repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.budgeted_source_role_service \
  ELIFE_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/independent-source-confirmation-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_RESULTS
```
