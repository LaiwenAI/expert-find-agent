# Expand one role within a single recheck call

The [headroom audit](../role-recheck-opportunity-v1) found additional
proofs when one incorrect clause could be corrected outside its
approved set. Test a deployable expansion on the same eLife cohorts
without using the intended roles or target source identifiers.

Keep the original search and recheck whenever it issues an initial
proof or has an approved clause to examine. Otherwise rank the role
taxonomy per clause by the mean standardized ridge and prototype
scores. Compare two ways to find a viable added role.

- **Language-first top three.** Nominate the highest-ranked role
  among the top three outside the existing approved set. Then
  test whether it completes a proof.
- **Feasibility within top five.** Scan the fixed top five in
  language-score order and nominate the first unapproved role
  that completes a proof while retaining the other clause's
  entire original band. This variant was added after the
  language-first screen produced no extra calls.

If both clauses qualify, take the higher standardized score and
use the first clause as tie break. Give the reader the complete
language-ranked shortlist and a none option, including roles
that cannot issue a proof. No source labels or author information
enter the prompt. Issue only if it selects the one expansion and
its gap clears the cutoff. Each request uses at most one reader
call across the old and new paths. No call is added after an
earlier read.

Use the original zero confidence gap or set a top-five gap cutoff
on the earlier eLife development cohort. The latter maximizes
strict proofs under the 5% observed adverse-issued constraint on
pooled development views. It can disable expansion. Evaluate
separately on the held paper pairs and missing-source controls.
Also count how often the expansion proposal would yield a strict
proof if accepted. This measure depends on the real proposal
and is narrower than supplying the intended role.

## Results

The narrow language-first plan finds no feasible option in either
cohort. It makes no added calls and preserves original counts.
The top-five plan adds 32 calls on development and produces two
additional proofs without an additional adverse result. Its
selected cutoff is zero. This calibration does not distinguish
the unfiltered and calibrated top-five plans.

| Held clue and route | Strict proofs / 66 | Adverse / issued | Original calls | Added calls | Exact if proposal accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excerpt original | 42 | 6/48 | 32 | 0 | — |
| Excerpt language-first | 42 | 6/48 | 32 | 0 | 0 |
| Excerpt top five | 44 | 6/50 | 32 | 29 | 6 |
| Keywords original | 33 | 1/34 | 19 | 0 | — |
| Keywords language-first | 33 | 1/34 | 19 | 0 | 0 |
| Keywords top five | 34 | 1/35 | 19 | 23 | 5 |

Neither expansion emits a substitute on the 264 missing-source
views per clue type. The wider plan calls the model 52 more
times across 264 available-source views, yielding three added
strict proofs. Eleven proposals could have yielded a strict proof
if accepted. The reader confirms three. Thus the narrow rule
fails to reach any opportunity, while the wider rule is limited
by both proposal yield and the reader's acceptance of correct
proposals. Pooled adverse share with the wider plan is 7/85.
Local P95 is about 26.5–26.6 ms for excerpts and 21.9–22.3 ms for
keywords. The similar P95 comes with substantially more reader
work and gives no evidence of similar service capacity. These
results do not justify enabling the wider plan by default.

Both tasks in Slurm array 46003 on the ws3 RTX 5090 reproduce
all non-timing fields with reversed order. Per-request timings
charge embedding, retrieval, branch checks and the selected
model call. No transport or unstructured parsing is included.
The results reuse the defined 13 held article pairs and their
paired wording. [Full output](results.json)

## Reproduce

Restore the earlier and held eLife projections from their
published manifests. Use Slurm CUDA, a fresh output and the
frozen model revisions. Repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.bounded_role_expansion \
  DEVELOPMENT_PROJECTION HELD_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```
