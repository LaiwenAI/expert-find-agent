# Learn which expert a shared source supports

The [reviewer relevance comparison](../cof-relevance-weighted-routing-v1)
could use human judgments of reviewer suitability. An independent
judgment of whether the cited document supports the person requires a
different label. The [TREC 2006 Enterprise Track](https://trec.nist.gov/data/t15_enterprise.html)
provides both grades for experts and grades for the topic, expert,
and document triple. Follow the [track definitions](https://trec.nist.gov/pubs/trec15/papers/ENT06.OVERVIEW.pdf):
evidence grade 2 supports the person, grade 1 is neutral, and grade 0
counts against expertise.

Use the same historical candidate and document menus as the
[joint scheduling comparison](../trec-joint-owner-source-plan-v1).
Hold out each of the 49 topics from training and from the graph used
to build every menu. The training topic's labels are also masked when
building its own menu. Use the top 60 people and up to 16 associated
documents each. All learned routes receive the same seven historical
features. Fit the following scores.

| Score | Supervision |
| --- | --- |
| Expert rank | Descending historical expert rank score |
| Pointwise, full menu | Joint expert and source grade 2 gives observed credit. All other menu entries receive zero observed credit. |
| Pointwise, judged | Among grade-2 experts, distinguish supporting evidence from explicit grade-0 or grade-1 judgments. |
| Pairwise, same source | Within each training topic and document, compare a supporting pair with a neutral or negative pair. Both experts must have grade 2. Fit mirrored feature differences with the same regularized logistic solver. |

Use two evaluations. The first scores ordering in every same-source
contrast among grade-2 test experts with known conflicting labels.
Ties are reported separately. The grade-0 subset isolates counterevidence.
Several pairs can share a topic or document, so the comparisons are
descriptive.

The second uses the fixed top 20 experts. It compares all four
checking orders in two panels with fixed document selections.

- Rank sum opens three documents prioritized by the sum of reciprocal
  expert ranks across their menu entries.
- Expected support opens three documents prioritized by the sum of
  their top two full-menu pointwise scores. This panel was added after
  inspecting the rank-sum result on the same leave-one-topic-out topics.

Within each panel, all routes open the same documents and use a
matched check budget. The original comparison allows up to two checks
per source. The one-check sensitivity fixes the realized budget across
both document selection rules as well. All comparisons reuse the same
document lists chosen before checking any evidence.
Count relevant experts with at least one checked supporting document
once, up to five per topic. Count neutral, negative, and unknown checks
separately. Also report the number of relevant people with a supporting
document anywhere in each selected source set, capped at five per
topic. This upper bound ignores the check quota. Held judgments assign
credit after each route selects its checks. The replay does not read
raw document text or time a service.

## Results

Of 49 held topics, 14 have reachable same-source contrasts in the
60-person menus. There are 46 contrasts, including five against
grade-0 evidence. All four scores resolve every contrast without ties.

| Score | Correct of 46 | Correct of five grade-0 contrasts |
| --- | ---: | ---: |
| Expert rank | 28 | 3 |
| Pointwise, full menu | 15 | 1 |
| Pointwise, judged | 14 | 3 |
| Pairwise, same source | 31 | 4 |

The two document rules expose known support for two and 13 distinct
topic–expert slots, respectively, before applying the check quota.

| Opened documents | Expert checks | Expert rank | Pointwise, full menu | Pointwise, judged | Pairwise, same source |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rank sum, 147 reads | 147 | 1 | 1 | 1 | 1 |
| Expected support, 147 reads | 147 | 5 | 4 | 4 | 4 |
| Rank sum, 147 reads | 259 | 1 | 1 | 1 | 1 |
| Expected support, 147 reads | 278 | 6 | 6 | 5 | 6 |

Cells count confirmed slots. At one check per source, the
expected-support selector confirms five slots with the historical
expert order, compared with one at the same 147 reads and 147 checks
for rank-sum selection. This is +1.63 percentage points of observed
supported P@5 in the 49-topic archive. The source-selection result
is exploratory because the second document rule and the one-check
comparison were added after earlier results on the same topics.

Same-source supervision improves descriptive ordering in the pooled
contrasts, but yields four of the same-budget slots where expert rank
yields five after expected-support selection. It ties expert rank
with two checks per source. The unchanged rank is a stronger next
control for prioritizing checks. Confirm source-selection gain on an
independent collection and compare to the existing dynamic scheduler
at the same realized budgets before adopting the document rule.
Of 147 checks with expected-support selection and expert rank, 119
have no judgment. The counts therefore measure confirmed archive
support, with unknown checks reported separately.

Slurm jobs 46846_0 and 46846_1 on ws3 reproduce the aggregate and
per-topic outcome signature in opposite topic orders. The final run
also reproduces the shared comparisons from the two pilot runs.

## Reproduce

Restore the official topics and qrels with the pinned digests in
[the results](results.json). Run both orders on a Slurm allocation.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.trec_enterprise.source_owner_contrast TREC_DATA NEW_RESULTS
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.trec_enterprise.source_owner_contrast TREC_DATA NEW_REVERSE_RESULTS --reverse
```
