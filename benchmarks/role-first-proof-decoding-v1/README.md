# Resolve the requested role before checking contributor membership

The [full-catalog proof study](../held-open-proof-search-v1) exposed a
citation error that owner-only scoring missed. Its ridge rule picks the
nearest declared role for each linked author. This study holds the two
selected papers fixed and instead decodes each role over all 14 CRediT
classes before examining any author's declarations. Exact membership on
both selected papers is required; the two declarations must name the
same resolved author. All decoders use one shared Qwen3-Embedding-0.6B
encoding of the role phrases. A different source route or a second model
call cannot explain differences between decoders in this replay.

Compare the archived author-conditioned ridge cutoff, the stricter
author-conditioned calibration, and calibrated ridge/definition
agreement. Test query-first ridge and a mixture of ridge, prototype,
and definition scores, each with and without a confidence cutoff.
Before averaging the banks, center and scale each phrase's scores over
the full taxonomy. Select mixture weights by six-fold out-of-fold
accuracy on the existing 84 training phrases, holding one phrase per
class out in each fold. Select on a quarter-step simplex. Ties favor
fewer active banks and then higher ridge weight. The query-first
confidence is the smaller of the two top-versus-runner-up gaps.
Calibrate each threshold on the same 30 eLife development paper groups
using supplied correct sources and allowing at most one adverse
development group. The development and held groups are source- and
recorded-author-disjoint. The same NISO wording inventory may be reused
across them, so this is neither a new wording holdout nor a bound on
deployment risk.

For eLife, keep the archived depth-six linked-source route over 2,011
papers, with excerpt and keyword clues scored separately. Replay
with either designated paper withheld as a source substitution check.
A strict proof must return the designated sources, a gold author,
and both exact requested labels. A request with any unsupported
output counts as wrong, even if it also returns a correct proof.
Separate wrong source selection, abstention on correct sources,
wrong owner only, and correct owner with wrong citation roles in
the original path. Compare global role predictions within the last
category.

For PLOS, reuse the frozen source-swap and role-edit panels from the
[original-text transfer study](../plos-selective-review-holdout-v1).
Provide the designated papers and reuse the eLife deterministic
selection of NISO phrases, based on the paper pair, case ordinal,
and role. The PLOS wording assignment and strict role metric differ
from the previous transfer report. The two PLOS panels share five
paper pairs and are reported separately. No parameters are fit on
the PLOS panels.

## Results

| Decoder | eLife excerpt, both present | eLife keyword, both present | PLOS source swaps | PLOS role edits |
| --- | ---: | ---: | ---: | ---: |
| Archived author-conditioned ridge | 57 / 40 | 50 / 34 | 9 / 4 | 18 / 7 |
| Recalibrated author-conditioned ridge | 21 / 4 | 20 / 4 | 2 / 0 | 5 / 0 |
| Calibrated ridge/definition agreement | 15 / 1 | 13 / 1 | 4 / 0 | 6 / 0 |
| Query-first ridge, open cutoff | 46 / 29 | 41 / 24 | 8 / 3 | 15 / 4 |
| Query-first ridge, calibrated | **23 / 0** | **21 / 0** | 3 / 0 | **7 / 0** |
| Query-first fitted bank, open cutoff | 53 / 26 | 46 / 20 | 8 / 1 | 16 / 4 |
| Query-first fitted bank, calibrated | 15 / 0 | 15 / 0 | 3 / 0 | 5 / 0 |
| Positive / total requests | 89 / 178 | 89 / 178 | 14 / 28 | 27 / 54 |

Cells report correct strict proof requests / wrong-output requests.
The calibrated query-first ridge improves the observed eLife operating
point relative to strict author-conditioned ridge, raising supported
counts by two excerpts and one keyword while removing four wrong
outputs in each view when both sources are present. On the two small
PLOS panels it stays at zero wrong and changes strict coverage by
+1 and +2 against that baseline. Calibrated bank agreement retains
one more PLOS source-swap proof at the same observed error count.
With a paper withheld, the eLife keyword route still issues two
unsupported requests out of 356 from one substituted paper group.
All eLife excerpt routes issue zero on the 356 withheld-source cases.
The calibrated query-first path covers just 21–23 of 89 positives
per eLife entrance, with the missing-source error concentrated in a
single group.

Among the original route's 32 excerpt and 39 keyword positives
without a strict proof, 21 and 19 respectively already had a correct
author with incorrect cited labels. Global ridge still mislabels at
least one role in every such request. The fitted bank decodes both
roles correctly in five requests in each view. Remaining failures
are missing source pairs (0 excerpt, 11 keyword), no proof on the
correct pair (10 and 9), and wrong identity alone (1 and 0).
The fitted weights select the prototype bank alone. It recognizes
80/84 out-of-fold training phrases versus 79/84 for ridge, and both
recognize 30/43 distinct held eLife NISO phrases. Selection on authored
phrases therefore gives little evidence of better role interpretation.
At the calibrated operating point, prototype confidence rejects more
correct proofs than ridge.

Both RTX 4090 jobs in array 47090 on ws1 agree on all quality counts,
cutoffs, and phrase scores reported here. The experiment isolates
decoding and does not time the changed policy online. The observed
coverage/error changes use paired requests with recurring wording
and few independent groups. They do not establish population
noninferiority or zero operational risk. Keep calibrated query-first
ridge as a control, test role-paraphrase generalization with
unseen wording, and evaluate source acceptance separately under
withheld-source negatives. [Aggregate results](results.json).

## Reproduce

Restore the eLife projections as in the
[full-catalog proof study](../held-open-proof-search-v1) and the PLOS
projections as in the [transfer study](../plos-selective-review-holdout-v1).
On an allocated GPU compute node, use a new output path:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.role_first_proof_decoding \
  CONFIRMATION_PROJECTION EXTENDED_PROJECTION \
  benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json \
  benchmarks/held-open-proof-search-v1/results.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  QWEN_EMBEDDING PLOS_SWAP_MANIFEST PLOS_SWAP_NOTES \
  PLOS_EDIT_MANIFEST PLOS_EDIT_NOTES NEW_RESULTS
```

Repeat with `--reverse` and a fresh output path. The script checks the
held eLife counts against all three archived controls before
writing new results.
