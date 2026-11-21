# Stress-test frozen contributor decisions with new role wording

The [role-first replay](../role-first-proof-decoding-v1) found a favorable
eLife operating point when it decoded the requested role before
checking the selected author's declarations. The calibration and test
there could share phrases from the same NISO inventory. This study
freezes the calibrations and changes only role wording. Keep the
2,011-paper catalog, publication clues, linked-source proposals,
30 held paper groups, and exact source/author/role contract.

Use 56 new English phrases, four per CRediT role. Two are concise
descriptions of the requested work. Two share the setup and subject
matter with a confusable neighboring role while changing the action
that establishes the intended responsibility. For example, the
design role defines a procedure for a weld study; the conceptual
role formulates its central question. The fixture covers seven role
boundaries and is inspectable in [phrases.json](phrases.json).
The project authored and labeled the phrases from the taxonomy
without inspecting model predictions. The script rejects normalized
duplicates of the 84 fit phrases, the archived wording inventories,
and other new phrases. This is a constructed stress set without
external annotation.

For each wording style, replay two parallel variants of every
original request. The lane determines which paraphrase of each role
is used. Each lane has 89 positives and 89 negatives per entrance,
plus 356 source-withheld requests. Lanes share paper groups and
reuse the same 28 phrases within each style. Treat these as paired
sensitivity analyses and avoid pooling their totals as independent
observations. A request may count as both correct and wrong if it
issues multiple proofs. Source substitutions always count as wrong.

Compare the archived strict author-conditioned decoder and calibrated
bank agreement against the query-first linear ridge and prototype
decoders. Add a Gaussian-kernel ridge control over the original 84
training vectors. The kernel uses cosine distance between normalized
vectors and diagonal regularization 0.1. Select its inverse bandwidth
from 1, 4, and 16 by holding one phrase per role out in each of six
folds. Break ties toward the smallest inverse bandwidth. Fit the
confidence threshold on the original NISO development groups with
the same adverse-group rule. All methods score the same embedding
vectors. The kernel adds a scan of 84 precomputed vectors per role
phrase and makes no additional model call. No threshold or model
selection uses the new wording fixture.

## Results

| Decoder | Terse exact roles / 28 | Shared-context exact roles / 28 | Terse excerpt proofs / 89 | Shared-context excerpt proofs / 89 | Terse keyword proofs / 89 | Shared-context keyword proofs / 89 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict author-conditioned | — | — | 63–72 | 12–38 | 57–64 | 9–33 |
| Calibrated bank agreement | — | — | 31–43 | 18–28 | 29–40 | 17–24 |
| Query-first linear ridge | 28 | 27 | 73–75 | 32–38 | 64–66 | 30–34 |
| Query-first prototype | 28 | 24 | 38–89 | 27–40 | 31–78 | 25–35 |
| Query-first Gaussian-kernel ridge | 28 | 25 | 63–75 | 27–38 | 54–66 | 25–34 |

The proof range spans the two paraphrase lanes with both sources in
the catalog. All methods emit zero wrong proofs on these constructed
lanes with both sources present. Linear ridge matches or improves on
strict author-conditioned counts in all eight lane/entrance
combinations, with the largest gains in the first shared-context lane.
Prototype coverage varies substantially between two concise
rephrasings of the same tasks. The additional kernel computation does
not improve the worst-lane count and there is no measured latency
benefit. The selected inverse bandwidth is 16 at 80/84 out-of-fold
accuracy, one phrase ahead of either smaller setting. On the original
NISO replay, kernel ridge emits one present-source mistake in each
entrance; the linear ridge control emits zero.

The before-cutoff counts separate confidence loss from semantic error.
With excerpt clues in the first shared-context lane, linear ridge
finds 89 exact proofs and zero errors before calibration; the frozen
cutoff retains 32. In the other shared-context lane it finds 76 exact
proofs and four wrong outputs before the cutoff, then retains 38 and
removes all four errors. With keyword clues the corresponding
correct counts are 78 to 30 and 67 to 34. A single softer cutoff
would need validation across both wording styles and missing-source
cases.

In the keyword entrance with a source removed, linear ridge emits
four unsupported proofs out of 356 requests in each new lane, up from
two with NISO phrases. Every such output comes from the same paper
group. Clearer responsibilities can increase the likelihood of
accepting a substitute source, even as they help on requests whose
sources are present. This is a targeted perturbation of the same
source groups. Its results motivate testing independent source
acceptance with additional blocked or missing-source cases.

Both RTX 4090 jobs in Slurm array 47110 on ws1 reproduce the
reported counts, frozen cutoffs, and wording confusions. The
diagnostic run also reproduces the calibrated outcomes from the
initial array 47104. [Aggregate results](results.json).

## Reproduce

Restore the published inputs from the
[role-first replay](../role-first-proof-decoding-v1).
Run on an allocated GPU compute node with a fresh output path:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.role_wording_stress \
  CONFIRMATION_PROJECTION EXTENDED_PROJECTION \
  benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json \
  benchmarks/role-first-proof-decoding-v1/results.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/role-wording-stress-v1/phrases.json QWEN_EMBEDDING NEW_RESULTS
```

Repeat with `--reverse` and a different output path. Before evaluating
new phrasing, the script checks frozen cutoffs and every original
control's strict, wrong, and adverse-group counts.
