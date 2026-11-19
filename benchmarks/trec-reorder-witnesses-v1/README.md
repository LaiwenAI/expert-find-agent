# Reorder supporting-source checks while certifying expert selection

Reuse the pinned TREC 2006 expert and evidence judgments and the 12 seeded orders per topic from the [canonical-citation check bound](../trec-check-lower-bound-v1/README.md). Keep ten candidates, their initial ranks and the two-source-prefix scheduler fixed. Relax the output contract so any sufficient set of verifier-accepted citations is admissible. For every replay, verify the exact same ordered expert IDs and promotion status against exhaustive verifier decisions. Under injected verifier errors, separately check the returned citations against the original judgments.

Compare the existing seeded source order, placing sources associated with more current candidates first, and placing sources with higher smoothed rates of grade-2 association on other topics first. The last policy uses no judgments from the held-out topic. It can use judgments for the same source on other topics, which occurred for 15,861 of 20,051 candidate–source associations. Both policies break ties using the seeded source order. The original order is a controlled baseline, not a production model ranking.

For an expert above the final eligibility boundary, a certificate requires the requested number of positive checks if promoted. Otherwise it needs enough negative checks to make promotion impossible. Summing those counts gives a label-aware lower bound on checks under unrestricted source ordering and independent inspection of each association. Counts under a simulated verifier use its fixed per-association responses. The bound does not assume joint reasoning about correlated model errors.

| Required citations, perfect verifier | Ordered by | Checks | Unique reads | Waves | Fully supported P@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Seeded control | 226.85 | 200.70 | 62.24 | 46.53% |
| 1 | Shared sources first | 226.78 | 190.64 | 62.01 | 46.53% |
| 1 | Other-topic source prior | 233.81 | 205.04 | 63.40 | 46.53% |
| 2 | Seeded control | 259.01 | 226.84 | 70.03 | 42.45% |
| 2 | Shared sources first | 259.19 | 215.39 | 70.12 | 42.45% |
| 2 | Other-topic source prior | 265.24 | 230.31 | 71.81 | 42.45% |

The flexible-citation lower bounds are 199.16 and 225.24 checks respectively. Shared-first reduces reads by about 5% under the perfect verifier; the paired, topic-resampled 95% intervals for checks saved include zero. The historical source prior does not close the gap. It adds 6.23 checks when two witnesses are required (95% paired interval 0.42 to 12.58 additional checks). With simulated independent verifier errors, fully supported P@5 is 36.94% for the control and 36.80% for shared-first when two witnesses are required. Exact expert-order agreement can therefore coexist with a change in citation quality.

These replays use the judged candidate and evidence pools. They expose no original source text, use no live model, and measure work rather than latency. Each query–document read can be shared across candidate owners, while attribution checks are charged per association. The result only licenses flexible citations if the application accepts different supporting sources. [Aggregate results](results.json)

Run inside a Slurm CPU allocation using the pinned qrels.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m experiments.trec_enterprise.reorder_witnesses \
  QRELS_DIRECTORY NEW_RESULT_JSON
```
