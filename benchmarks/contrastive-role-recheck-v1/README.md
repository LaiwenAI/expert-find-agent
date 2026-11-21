# Compare confusable roles before issuing a linked proof

The [issuance audit](../role-issuance-audit-v1) found that each
reader-path error in the reused eLife holdout involved the revised
role. Test a choice between two candidate interpretations of that
role. Keep source retrieval, selection of the clause to review,
initial role bands and the independent proof check fixed.
The role reader has access to the request phrase and the role
options. It never sees which role yields an eligible contributor.

The existing reference uses all role names within the ridge band.
The test arms use the two roles in that band with the highest mean
standardized ridge and prototype scores. Compare the following
renderings under Qwen3-4B-Instruct-2507.

- **Pair names:** Show the two names and the none option.
- **Pair contrast:** Give each role a short criterion and its
  delineation from related actions. Include an explicit neither
  or indeterminate choice.
- **Pair stable:** Batch the contrast prompt and its reversed
  option order. Accept a role only if both orders select it.
  Use the smaller of the two confidence gaps.

For each pair route and the full role-name list, choose a confidence gap cutoff on the earlier
[280-article development panel](../elife-one-role-transfer-v1).
Pool excerpt and keyword views with both sources visible. Maximize
strict proofs among cutoffs with at most 5% observed adverse issued
results. Also compare both name lists at the original cutoff to
separate list length from calibration. Apply
the selected thresholds to the separate
[393-article held cohort](../elife-prompt-holdout-v1).
The held cohort was used for the initial error diagnosis, so this
is a follow-up test on known data with disjoint calibration
sources. Both cohorts reuse the NISO wording bank.

## Results

| Reader path | Excerpt proofs / 66 | Excerpt adverse / issued | Keyword proofs / 66 | Keyword adverse / issued | Excerpt local P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All names, original cutoff | 42 | 6/48 | 33 | 1/34 | 26.4–26.5 ms |
| All names, calibrated | 42 | 6/48 | 33 | 1/34 | 26.4–26.5 ms |
| Pair names, original cutoff | 41 | 8/49 | 33 | 1/34 | 25.5–25.9 ms |
| Pair names, calibrated | 41 | 4/45 | 33 | 1/34 | 25.5–25.9 ms |
| Pair contrast, calibrated | 36 | 5/41 | 29 | 1/30 | 25.4–25.9 ms |
| Pair stable, calibrated | 35 | 3/38 | 28 | 1/29 | 29.7 ms |

The development cutoff for all names is zero, so separate
calibration retains its original behavior. Shortlisting alone
increases excerpt errors from six to eight at the shared zero
cutoff. The pair-specific calibrated gap of 3.5 removes four
of those eight errors without losing a proof relative to its
unfiltered pair route. It leaves 74 strict proofs and five
adverse results among 79 issued views across excerpt and
keyword clues. The reversed-order check issues fewer adverse
outcomes in absolute terms, with 63 proofs and four adverse
outcomes among 67 issued views. No route meets the development
risk target on the held pooled views.

Each reader route makes 32 excerpt and 19 keyword calls. The
calibrated pair-names route uses 2,260 excerpt input tokens,
compared with 3,074 for all names. The order-stable route uses
6,224 tokens across both orderings and raises local P95. There
are no substitutions on the 264 missing-source views per clue
type. Additional role-boundary text and position agreement
cannot correct stable semantic confusion in this fixture. The
shortlist and its confidence cutoff are separate choices and
need independent risk and coverage validation.

Both tasks in Slurm array 45987 on the ws3 RTX 5090 reproduce
all non-timing fields in reverse order. Timing charges the live
embedding, anchored retrieval, proof checks and the measured
reader path, with separate one-prompt and two-prompt calls for
fair comparison. It excludes transport and unstructured
request parsing. Repeated roles and 13 held paper pairs limit
the risk inference. [Full results](results.json)

## Reproduce

Restore the earlier eLife projection and the newer held
projection from their pinned manifests. Use the same model
revisions and frozen role artifacts. Run on Slurm CUDA with a
fresh output path, and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.contrastive_role_recheck \
  DEVELOPMENT_PROJECTION HOLDOUT_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```
