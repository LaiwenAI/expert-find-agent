# Change the requested facet while holding both articles fixed

Use the [CSFCube expert-adjudicated labels](https://openreview.net/forum?id=8Y50dBbmGU).
Group judgments by query paper and candidate paper and keep pairs with
two judged facets. The query paper and candidate paper remain identical
when the requested scientific aspect changes. The main comparison uses
all 101 pairs with grade 0 for one facet and grade 2–3 for the other.
Retain 111 grade-1 versus grade-2–3 pairs separately, all 42 pairs whose
two facets are strong, and up to 12 hash-sampled two-grade-0 controls
per query paper (192 total). No unjudged facet is introduced.

Compare a generic whole-article reader, an explicit facet prompt with
full abstracts, the same prompt with up to three facet-tagged sentences
on the query side or both sides, a sentence-count-matched lead
control on both sides, and cosine using both facet excerpts.
Lengths in tokens can differ at the same sentence count.
The released sentence labels are model predictions, not the evaluation
judgments. An empty facet excerpt uses the first available sentence.
Use frozen Qwen3-0.6B encoder and reranker checkpoints. For the
contrastive arm, subtract a nonnegative weight times the competing
facet's full-input logit. The second evaluated facet is provided by
this paired benchmark. This contrasts fit and specificity and does not
define how to choose a competing facet for arbitrary requests.

## Evaluation

Hash-split the 16 paired query papers into four held-out folds.
For each fit, also remove any row involving an article present in
that fold, whether as query or candidate. Fit one strict cutoff with
at most 10% grade-0 acceptance on the sampled fit pairs. For the
contrastive route, fit a weight from 0, 0.25, 0.5, and 1 before
holdout. Prefer correct grade-0/strong switches, then more strong
acceptances, then the smaller weight. The empirical operating target
does not guarantee the held grade-0 rate.

Report whether the strong facet has a strictly greater raw score,
as well as whether the calibrated decision accepts the strong facet
and rejects its opposite. Report both-strong acceptance to detect
over-suppression, and grade-0/1/strong decisions separately.
Pair the switch-rate comparison against the full-input explicit and
matched-lead controls. Intervals cluster by query paper. The sample is a conditional
diagnostic on already judged pairs. It does not measure how many
experts can be found, the author's personal contribution, or serving
latency. The prior [CSFCube source-retrieval study](../csfcube-facet-challenger-v1)
changes which paper is read. This experiment holds papers fixed.

## Results

The controlled Slurm array `45744` runs on the ws3 RTX 5090.
The 446 pairs come from 16 query papers. Each article-disjoint fit
has 304–332 pairs. The first result column counts both accepting
the strongly relevant facet and rejecting the grade-0 facet.

| Route | Correct switches / 101 | Grade-0 accepted / 485 | Strong accepted / 296 | Both strong accepted / 42 |
| --- | ---: | ---: | ---: | ---: |
| Generic | 0 | 45 | 157 | 33 |
| Explicit facet, full abstracts | 5 | 45 | 155 | 30 |
| Query-side excerpt | 9 | 46 | 153 | 29 |
| Both-side matched leads | 13 | 47 | 156 | 24 |
| Both-side facet excerpts | 21 | 43 | 176 | 31 |
| Facet cosine | 24 | 47 | 187 | 31 |
| Contrastive full logit | 18 | 50 | 120 | 17 |

Compared with the full explicit prompt, both-side facet excerpts
gain 15.84 percentage points in correct switches (paired 95%
interval +6.74 to +24.74), and facet cosine gains 18.81 points
(+9.38 to +26.96). The comparison with the matched lead is less
certain. Both-side facet excerpts gain 7.92 points (−2.82 to +16.54),
and facet cosine gains 10.89 points (−2.25 to +20.88). Equal sentence
counts can leave token-count differences. These controls do not
isolate the details of rhetorical segmentation. The facet cosine
score is a candidate for a cheap gate before cross-encoder
verification, subject to a live and target-task comparison.

The contrastive route gains switch accuracy and loses both-strong
coverage: it jointly accepts only 17 of 42 both-strong pairs
versus 30 for the full prompt. A competing facet should not be
treated as a negative unless the task requires exclusivity.
The correct-switch rates also leave most difficult requests
unresolved at these operating points. No conclusion establishes
expert match quality or first-query speed.

[Aggregates and paired intervals](results.json).

## Reproduce

Use the [four pinned source files](https://github.com/iesl/CSFCube/tree/7ffe012c1d92c470f185fc086086caf5b8e05ef3)
and verified open checkpoints inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.csfcube.paired_facet_confirmation \
  CSFCUBE_DIR benchmarks/csfcube-lexical-challenger-v1/provenance.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
