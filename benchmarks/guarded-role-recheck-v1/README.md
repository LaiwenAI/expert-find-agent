# Guard source and role evidence in single-clause rechecks

Reuse the engineering and [eLife](../elife-one-role-transfer-v1)
panels. Both include a scope-swap negative for each positive and
separate controls that hide either target source. The anchored
join chooses the source pair before any role decision. The
strong static route applies the previous prototype band. The
recheck route uses the previous ridge band, selects one ambiguous
clause by its ability to complete a same-author proof, and calls
Qwen3-4B only where a single-clause answer could change the output.

Apply two source checks independently and in combination with
one role check. The checks apply only to the additional recheck,
so an already supported ridge proof is kept.

- **Two leaders.** Require each selected article to lead its own
  fused retrieval list. A linked pair cannot pass this check by
  borrowing a strong match from just one side.
- **Calibrated source.** Refit the exact earlier PLOS rules on the
  same pinned 24 probes, using separate excerpt and keyword gates.
  Require both selected articles to pass their own source checks.
  There is no alternative pair search after rejection.
- **Role agreement.** A narrowed role must also occur in the
  selected clause's frozen prototype band. If none of the
  overlapping roles could complete a proof, avoid the model call.
  The reader still sees its original ridge options to avoid
  forcing a compatible answer.

The existing engineering and eLife cohorts were reused to
diagnose previously observed errors. The cutoff, role bands and
source fitting procedure are frozen from previous studies.
Neither target cohort fits a parameter in this experiment.

## Results

In eLife, there are 51 positives, 51 scope swaps, and 204
missing-source views for each clue type. The table reports
strict positive proofs, adverse outputs with both sources
visible, and substitutions when a target is hidden.

| Role/source policy | Excerpt proofs / adverse / substitutions | Keyword proofs / adverse / substitutions | Excerpt / keyword reader calls |
| --- | ---: | ---: | ---: |
| Static prototype band | 23 / 0 / 0 | 16 / 0 / 0 | 0 / 0 |
| Recheck | 32 / 3 / 2 | 25 / 3 / 0 | 33 / 25 |
| Role agreement | 30 / 0 / 0 | 23 / 0 / 0 | 22 / 17 |
| Two source leaders | 32 / 3 / 2 | 25 / 3 / 0 | 32 / 25 |
| Two leaders and role agreement | 30 / 0 / 0 | 23 / 0 / 0 | 21 / 17 |
| Calibrated source | 31 / 2 / 0 | 22 / 2 / 0 | 30 / 20 |
| Calibrated source and role agreement | 29 / 0 / 0 | 21 / 0 / 0 | 20 / 14 |

On eLife, the role agreement rule removes all observed adverse
outputs and source substitutions in these views, while dropping
two strict proofs per clue type from unguarded recheck. It can
reject the particular role supporting a wrong-source output.
It does not validate the source independently. Requiring two
fused leaders leaves the excerpt substitutions unchanged. The
PLOS-calibrated source check removes those substitutions, but
leaves two adverse eLife outcomes per clue type. Combining the
source check with role agreement reduces calls further at the
cost of additional positive proofs.

In the four-pair engineering panel, for each clue type, the
prototype band returns 4/15 strict proofs, recheck returns 6/15,
and role agreement returns 5/15. All three have zero observed
adverse outputs on this panel. With keyword clues, the calibrated
source check keeps just 2/15, both with and without role
agreement. The zero event counts are descriptive for these
reused panels, which have repeated requests within paper pairs.

On eLife excerpt queries, role agreement reduces mean local
latency from 16.8 to 15.2 ms relative to unguarded recheck.
P95 is 34.2 ms, compared with 35.2–36.1 ms for recheck and
12.8–12.9 ms for the static prototype band. For keywords, P95
is 29.6–29.8 ms with role agreement and 10.5–10.6 ms for the
static baseline. The keyword combined guard lowers this to
21.5–21.8 ms while dropping two more proofs. The results leave
a substantial tail-latency gap to close before a low-cost first
query claim.

Both tasks of Slurm array 45937 ran on ws3 RTX 5090 and
reproduced all non-timing fields with reversed request order.
The eLife baseline and recheck counts match the previous run.
Times factor the shared query embedding and source ranking into
each arm, then charge its guards and selected reader call.
Local timing excludes index construction, network transport and
unconstrained request parsing. [Full results](results.json)

## Reproduce

Restore the pinned eLife, engineering and PLOS projections
from their linked benchmarks. Reuse the previous engineering
search bands and one-role margin. Use a fresh result path
inside a Slurm CUDA allocation and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.guarded_role_recheck \
  ELIFE_PROJECTION ENGINEERING_PANEL PLOS_PROJECTION NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```
