# Transfer source confirmation to shorter clues and engineering papers

Apply the exact [frozen eLife keyword cutoffs](../independent-source-confirmation-v1)
to a changed clue distribution and a source-disjoint engineering
corpus. Keep embedding and verifier model revisions, candidate
generation, role band, and acceptance rules fixed. Score three
renderings: up to eight original IDF-ranked keywords, the first
four, and the last four of the original list. Short views can
overlap when fewer than eight terms are available. These are
controlled changes in clue information, not independently
authored human queries.

The [393-article eLife holdout](../elife-prompt-holdout-v1) tests
wording transfer. The [53-article engineering panel](../engineering-cross-source-search-v1)
tests transfer to Frontiers and Sensors papers on four linked
article pairs. Both cohorts keep the original positive and scope-swap
requests. Score separate controls hiding each required paper.

The baseline keeps the previous top-six anchor. Ungated fallback
uses one author-linked proposal after the baseline abstains.
The one-side and two-side routes verify the same proposal with
the earlier source-only criteria. Report correct source pairs
and wrong-source pairs proposed for issuance before roles,
as well as strict prototypes and substitutions after roles.
Wrong-source pairs are deduplicated by target, selected pair and
availability condition to quantify repeat effects. No target data
selects a cutoff.

## Results

The frozen logits are −5.125 for one side and −5.1875
for two sides. They produce identical counts on both cohorts
and all three clue views. The table reports the one-side policy
and the comparison routes. Correct pair counts use positive
variants only. Static strict proofs additionally require the
intended roles and exact contributor set.

| Cohort and clue | Correct positive pairs, anchor / ungated / verified | Static strict proofs, anchor / ungated / verified |
| --- | ---: | ---: |
| eLife, original | 48 / 61 / 53 of 66 | 26 / 33 / 29 of 66 |
| eLife, leading four | 28 / 48 / 48 of 66 | 19 / 27 / 27 of 66 |
| eLife, trailing four | 25 / 35 / 35 of 66 | 14 / 20 / 20 of 66 |
| Engineering, original | 10 / 15 / 15 of 15 | 4 / 4 / 4 of 15 |
| Engineering, leading four | 7 / 7 / 7 of 15 | 1 / 1 / 1 of 15 |
| Engineering, trailing four | 5 / 11 / 10 of 15 | 0 / 4 / 3 of 15 |

On the 60 engineering missing-source variants per clue view,
the ungated fallback outputs 30, 26 and 48 wrong source pairs
before role matching. The frozen verifier reduces these counts
to 0, 0 and 4. In the trailing-four view, the four surviving
proposals repeat one wrong source pair. Three yield a substitute
with the intended roles supplied, and one still issues with the
static role bank. Source-level rejection and role-qualified
issuance therefore measure different effects. The verifier
also rejects all six wrong source pairs in the engineering
leading-four view with both target articles available. No route
has visible substitutions in any of the 264 eLife missing-source
variants per clue view.

The engineering source stage transfers on original keywords:
verification recovers five correct source pairs and rejects
all missing-source substitutes there. Its static strict proof
total stays at four, so the source improvement does not alone
fix role interpretation. Less distinctive keywords reintroduce
one wrong linked source and one actual substitute. The eLife
leading-four result supports a retrieval improvement under
that perturbation and retains one extra static adverse request
relative to the shallow baseline.

The one-side route has 18.2–18.6 ms local P95 on engineering,
compared with 19.4–20.0 ms for two sides. Both cost more than
the roughly 10 ms anchor. Both runs of Slurm array 46039 on
ws3 RTX 5090 reproduce non-timing results under reversed order.
Results reuse four linked engineering pairs and 13 eLife pairs.
Timings cover warm embedding, source selection and the live
reranker call, without transport or free-form parsing.
[Complete results](results.json)

## Reproduce

Restore the original eLife and engineering projections from
their manifests. Use the prior frozen score output and model
revisions. Run in Slurm CUDA with a fresh output path and repeat
with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.source_confirmation_transfer \
  ELIFE_PROJECTION ENGINEERING_PANEL benchmarks/elife-prompt-holdout-v1/manifest.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/independent-source-confirmation-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
