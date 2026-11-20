# Route first-query verification by its expected contribution

Reuse the [frozen CSFCube live setup](../csfcube-live-facet-gate-v1).
Every online route uses one query embedding and the same eight
full-abstract dense candidates. The fast route scores compact
candidate facets using the existing query vector and emits up to
two accepted papers in dense order. The fallback checks the four
highest compact-cosine candidates with the compact cross-encoder.
The calibration and shortlist match the earlier comparison.

Compare fixed dense four, fixed facet four and pure cosine with
four request-level fallback policies. Each adaptive policy targets
30% fallback on its fit fold. The live hash arm assigns fallback
using a stable hash. Margin prioritizes proximity to the cosine
cutoff. Disagreement prioritizes different top-four memberships
under whole-abstract and facet retrieval. The learned rule
predicts fallback minus fast-path utility with five input
features: fast-path issuance count, largest cosine margin,
nearest cutoff distance among the first four candidates, cosine
spread, and top-four membership disagreement.

Fit four query-paper folds and hold out the evaluation fold. Do
not fit on requests containing any query or selected candidate
article present in the evaluation fold. Train the ridge rule on
one point for any strong source returned and a half-point penalty
when the two slots include an adjudicated grade-0 source.
Grade-1 issuance is reported separately. Fit using existing
full-batch observations, then run policies live with four-check
verification. The small fit samples and the chosen utility
weight are part of this pilot.

For each live route, randomly assign the same number of actual
fallbacks within each fold in 2,000 replays. Each replay chooses
between the two observed live outputs on that request. This
checks allocation value at equal realized work. It does not
measure randomized latency. Report query-paper paired intervals
and return grades, plus warm local P50/P95 including query encoding,
candidate retrieval, routing, verification and serialization.
No queue or network timing is included.

## Results

The two tasks of array `45763` reproduce all non-timing results
on the ws3 RTX 5090. Fold fits include 34–39 eligible requests.
The data contain 50 requests from 34 query papers.

| Route | Strong-return queries | Issued grade 0 / grade 1 | Requests verified | Warm P50 | Warm P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense four | 26 | 8 / 13 | 50 | 21.29–21.43 ms | 23.42–24.28 ms |
| Facet four | 27 | 8 / 16 | 50 | 21.41–21.62 ms | 23.68–24.05 ms |
| Cosine | 27 | 7 / 15 | 0 | 8.37–8.54 ms | 8.57–8.74 ms |
| Hash fallback | 27 | 8 / 14 | 16 | 8.54–8.64 ms | 22.36–22.41 ms |
| Margin fallback | 25 | 5 / 14 | 14 | 8.51–8.59 ms | 22.97–23.04 ms |
| Disagreement fallback | 28 | 6 / 15 | 15 | 8.52–8.63 ms | 22.29–22.50 ms |
| Learned fallback | 32 | 10 / 19 | 17 | 8.51–8.65 ms | 21.67–21.87 ms |

Learned allocation raises strong-return coverage by 12 percentage
points relative to dense four (paired query-paper 95% interval 0
to +24.56 points) while reducing four-pair batches from 50 to 17.
It issues two more grade-0 papers and six more grade-1 papers.
The same-budget random replay produces 26–31 strong-return
requests in its central 95% range. This exploratory range is not
an independent confirmation of the learned rule. The margin
heuristic loses two strong returns against pure cosine.

The learned route's lower median follows from bypassing the
cross-encoder on most traffic. The tail still contains fallback
requests and improves only modestly. Preserve fixed dense four
and same-budget routing controls. A held-out corpus with independent
qualification judgments is needed to verify the quality–cost gain
and estimate adverse issuance. No author attribution is tested here.

[Results for both runs](results.json).

## Reproduce

Restore the same CSFCube inputs and weights as the earlier live
benchmark. Run under Slurm CUDA and repeat with the alternate
order offset.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.csfcube.facet_fallback_routing \
  CSFCUBE_DIR benchmarks/csfcube-lexical-challenger-v1/provenance.json \
  QWEN_EMBEDDING QWEN_RERANKER benchmarks/csfcube-live-facet-gate-v1/results.json \
  NEW_RESULTS --order-offset 0
```
