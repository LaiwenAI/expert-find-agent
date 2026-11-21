# Recover a second source through author postings

The [role opportunity audit](../role-recheck-opportunity-v1) found
18 of 66 held keyword positives with an incorrect or absent source
pair. Measure candidate exposure, then test a bounded author
lookup over the [eLife held catalog](../elife-prompt-holdout-v1).
The candidate policy never sees the request's role labels or
the target source identifiers.

Use the same reciprocal-rank fusion and the earlier source
anchor. The main baseline searches the top six on each side.
It requires one fused leader, a common published ORCID and both
document cosine scores within 0.04 of the corresponding dense
leader. Compare the same rule on depth 24. For the author
postings policy, keep the top-six result if it selects a pair.
Otherwise consider each clause's fused leader as an anchor.
Intersect its coauthors' publication postings with the top 24
of the opposite clause. Accept a partner only if it satisfies
both a dense-gap limit and a BM25 score fraction relative to
that clause's lexical leader. Compare a fixed 0.04 dense-gap
check on the anchor with a symmetric gate requiring the same
dense and lexical conditions on both sides. The symmetric
variant follows the initial finding that most missed pairs
were already in the shallow pools. Among passing partners, use
the best fused rank and a deterministic source ID tie break. A diagnostic
fallback removes both acceptance checks and keeps the author link.

Choose a dense gap from {0.04, 0.08, 0.12} and lexical fraction
from {0.25, 0.50, 0.75} separately by clue type on the
[280-article development catalog](../elife-one-role-transfer-v1).
Maximize exact proofs when roles are supplied, then pair
accuracy, subject to no more missing-source substitutions
than the shallow baseline on development. Freeze both settings
for held evaluation. Count sources, strict proofs with the
frozen prototype role band, and strict proofs with the intended
roles supplied as a source-only diagnostic. Keep the latter
separate from the executable end-user result.

## Results

On the 66 held keyword positives, both intended sources appear in
the top six for 61 and in the top 24 for the same 61. Every pair
has at least one fused leader. Only 48 pairs meet the two 0.04
cosine checks. Therefore, 13 misses are due to the acceptance
rule even though both sources are present. The other five remain
unreachable at depth 24. The calibrated two-signal methods both
choose the 0.04 cosine limit and 0.75 BM25 fraction. Neither
finds more proofs than the shallow baseline on development.

| Keyword policy | Dev correct source pairs / 51 positives | Dev missing-source substitutions with oracle roles / 204 | Held correct source pairs / 66 positives | Held static strict proofs / 66 | Held static adverse / 132 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-six anchor | 36 | 0 | 48 | 26 | 1 |
| Top-24 anchor | 36 | 0 | 48 | 26 | 1 |
| Author postings, calibrated | 36 | 0 | 48 | 26 | 1 |
| Symmetric postings, calibrated | 36 | 0 | 48 | 26 | 1 |
| Author postings without topic acceptance | 36 | 5 | 61 | 33 | 2 |

On held keywords no policy issues substitutes on 264 missing-source
views. However, the ungated route issues one wrong-source result
with the static role bank on the development missing-keyword views,
and five with intended roles supplied. On development excerpt
controls it issues 11 wrong-source results with intended roles,
compared with one from the baseline. All held excerpt policies
already retrieve every correct pair. The positive keyword recovery
shows the author link can expose useful evidence. The differences
across missing-source cohorts show why the link alone is an
insufficient acceptance criterion.

Both runs of Slurm array 46018 on ws3 RTX 5090 reproduce the same
non-timing results under reversed case order. The source timing
charges the warm embedding batch and source selection. It excludes
a language-model recheck, free-form query parsing, index construction
and network transport. The search differs only in the source policy;
the prototype role bank is unchanged. Sources and wordings repeat
within each cohort. The held corpus contains 13 article pairs with
positive and swap controls. [Full results](results.json)

## Reproduce

Rebuild both eLife projections from the pinned publisher manifests.
Run inside Slurm CUDA with the frozen embedding and role-band inputs.
Repeat with `--reverse` and a fresh result path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.keyword_author_rescue \
  DEVELOPMENT_PROJECTION HELD_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json QWEN_EMBEDDING NEW_RESULTS
```
