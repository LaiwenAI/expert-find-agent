# Share one verification call across source and role decisions

Compose the previously tested [one-side source confirmation](../source-confirmation-transfer-v1)
and [name-only role recheck](../compact-role-prompts-v1). Freeze both
model revisions and cutoffs. Use the same eLife and engineering
catalogs, three keyword views, and missing-paper controls. Each
query gets one live embedding batch. Source and role verification
use Qwen3-Reranker-0.6B and Qwen3-4B-Instruct-2507 respectively.

The main comparison uses the original top-six source anchor and
static prototype role band. The source-only arm verifies an
author-linked fallback when the anchor abstains. The role-only
arm uses the proof-aware single-clause recheck on anchored pairs.
The combined arm enables both, and can call both models after a
source fallback. The role recheck retains its previously tested
ridge band and prototype agreement condition.

The one-call arm uses the same role path on anchored pairs. On
fallback pairs it spends the sole verification call on source
confirmation and uses the static prototype role band. A screened
variant checks whether that static band can produce any same-person
proof before spending the source call. If no proof is possible
under the remaining budget, it abstains without verification.
This screen depends on the declared role index and has no access
to target labels. Check equality of actual emitted source,
role sets and contributors against the unscreened one-call arm
for every request.

## Results

Held eLife, original keyword views, with both targets available:

| Policy | Strict proofs / 66 | Adverse / issued | Source calls / 132 | Role calls / 132 | Local mean ms | Local P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static | 26 | 1/27 | 0 | 0 | 10.6–10.8 | 10.9–11.1 |
| Source only | 29 | 1/30 | 26 | 0 | 12.3–12.5 | 19.3–19.4 |
| Role only | 33 | 1/34 | 0 | 19 | 12.4–12.6 | 21.9–22.2 |
| Both | 36 | 2/38 | 26 | 21 | 14.3–14.5 | 22.1–22.4 |
| One call | 36 | 1/37 | 26 | 19 | 14.1–14.4 | 21.9–22.3 |
| Screened one call | 36 | 1/37 | 9 | 19 | 13.0–13.2 | 21.9–22.2 |

The screen skips 17 of 26 source checks and preserves every
issued source, role and contributor set relative to one call.
Compared with unconditional composition, it preserves the same
number of strict proofs and has one fewer adverse outcome.
The two-call policy actually uses both models on two requests
in this view. The screened policy remains more expensive than
the static route and uses 19 language-reader calls.

The screened route also saves 31 source checks in the leading-four
eLife view. It keeps 29/66 strict proofs and one adverse result,
against 33/66 and one adverse result for both models. The local
P95 is about 19–20 ms rather than 30 ms. On trailing-four eLife
clues it keeps all 24 strict proofs of the combined route.
No eLife missing-source variant yields a substitute for any
tested route. The lower-information cases can benefit from
paying for both checks and require an explicit cost comparison.

On the original engineering keywords, screening skips all ten
source checks and keeps the six strict proofs and zero adverse
outputs of the combined route. With trailing-four engineering
clues, the one-call routes issue one substitute in 60 missing-source
variants; both models issue three. The screen does not repair
a source-only match error that still satisfies the static role
requirements. Conditions share paper pairs and role variants.
Counts should not be treated as independent field-risk estimates.

Both runs of Slurm array 46045 on ws3 RTX 5090 reproduce every
non-timing field when query order reverses. All measured paths
include the warm shared embedding batch, source selection and
their actual verifier actions. The paired benchmark factors
out shared inference. Its local timing excludes service queueing,
unstructured request parsing and index build. [Full results](results.json)

## Reproduce

Restore the pinned eLife and engineering projections. Reuse the
earlier bands and both frozen score outputs. Run inside Slurm
CUDA with a fresh path and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.budgeted_source_role_search \
  ELIFE_PROJECTION ENGINEERING_PANEL benchmarks/elife-prompt-holdout-v1/manifest.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/independent-source-confirmation-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_RESULTS
```
