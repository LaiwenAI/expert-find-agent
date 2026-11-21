# Test transferred reviewer relevance in evidence scheduling

Reuse the pinned depth-12 [CoF captures](../cof-deep-source-routing-v1)
for 122 KDD and 59 SciRepEval requests. All routes keep the same
five experts, their order, their source inventories and the
source verifier's frozen title-and-abstract decisions. The first
wave opens the union of each owner's first three sources. A
citation package requires two accepted sources.

Fit on the other collection after removing training rows whose
normalized title occurs in the target collection. Each learned
route uses the original source acceptance model. For relevance,
learn a separate owner score from rank, inventory and shallow
source features. Compare supervising on observed positive labels
against supervising only on judged positives and negatives.
Unknown grades supply zero observed credit in the first fit and
are excluded from the second.

| Follow-up policy | Owner utility |
| --- | --- |
| Dense reference | Unit utility with the frozen dense state policy |
| Learned acceptance | Unit utility with the transferred source acceptance model |
| Rank prior | Reciprocal expert rank with the same acceptance model |
| Observed relevance | Transferred owner model fit over all candidate slots |
| Judged relevance | Transferred owner model fit on judged slots only |

The three weighted policies map the highest owner score per
query to one and use a 0.25 floor. All routes use no more follow-up
reads and input tokens than the realized dense reference for that
query. The reference permits up to five follow-up reads. Different
routes may leave different amounts of that budget unused.

Match issued counts at the minimum per query across all routes.
Preserve every citation package established in the first wave.
For newly completed packages, keep expert rank order when
matching output counts. As a sensitivity analysis, report
matching by rank without that preservation as well. CoF labels
reviewer relevance. Relevance counts among model-cited slots
do not independently validate the content or ownership of a citation.

## Results

| Panel | Follow-up policy | Reads | Tokens | Untrimmed known positives / cited | Matched positive / judged negative |
| --- | --- | ---: | ---: | ---: | ---: |
| KDD | Dense reference | 610 | 320,095 | 12 / 291 | 12 / 44 |
| KDD | Learned acceptance | 600 | 296,998 | 12 / 302 | 12 / 44 |
| KDD | Rank prior | 597 | 297,472 | 12 / 302 | 12 / 44 |
| KDD | Observed relevance | 599 | 298,172 | 12 / 301 | 12 / 44 |
| KDD | Judged relevance | 598 | 296,403 | 12 / 302 | 12 / 44 |
| SciRepEval | Dense reference | 295 | 127,147 | 26 / 105 | 25 / 21 |
| SciRepEval | Learned acceptance | 290 | 119,939 | 29 / 111 | 25 / 22 |
| SciRepEval | Rank prior | 289 | 120,379 | 29 / 108 | 25 / 22 |
| SciRepEval | Observed relevance | 292 | 120,022 | 28 / 106 | 25 / 21 |
| SciRepEval | Judged relevance | 292 | 120,087 | 28 / 106 | 25 / 21 |

The matched totals are 288 cited slots for KDD and 99 for
SciRepEval per policy. The learned relevance weights do not
increase the known positive count on either panel. The additional
known positives in the untrimmed SciRepEval outputs accompany
additional issuance. Rank-only matching also gives no gain for
the learned relevance weights.

On the 56 within-query SciRepEval comparisons with one judged
positive and one judged negative, reciprocal rank orders 39
correctly. The two transferred owner models order 16 and 17
correctly. KDD has only 16 such comparisons, with counts of four,
three and two. This diagnostic applies to paired judged slots
and does not estimate calibration under missing judgments.
The result gives a direct reason to retain the simple rank and
source-acceptance controls in inter-collection transfer.

Slurm job 46824 on ws3 reproduces all routing outcomes in both
directions of collection traversal. The dense reference reproduces
the prior five-read metrics. These are archive replays of frozen
GPU decisions. No live source access or service latency is measured.

## Reproduce

Restore or rebuild the two pinned captures as in the
[depth-12 comparison](../cof-deep-source-routing-v1#reproduce).
Run on a Slurm allocation, then repeat with `--reverse`.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_relevance_weighted_routing KDD_CAPTURE SCIREPEVAL_CAPTURE \
  benchmarks/cof-deep-source-routing-v1/results.json NEW_RESULTS
```

[Aggregate results](results.json)
