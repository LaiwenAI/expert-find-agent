# Expand raw-text experience anchors at a controlled binding budget

The [raw-text experience study](../goja-raw-experience-scope-v1)
found only 224 of 318 gold anchor spans among its generated
candidates. Reuse its 199 fit ads, 115 development ads and 186
test ads from the [anonymized GOJA release](https://github.com/KruegerETRF/GOJA/tree/06c9887cceafdacec319ea222b16a4419870eca6).
The fit and test splits share neither identical ad text nor
annotations. Possible near-duplicate templates are outside the
deduplication check.

A diagnostic on the development ads found 49 absent gold anchors.
Thirty had an endpoint next to an alphanumeric character outside
the annotated span. Twelve had neither a cue nor a known surface
form, two spanned punctuation, and five had none of those flags.
Use these counts to distinguish word-internal boundaries from
the breadth of the semantic cue filter.

## Compared generators

| Generator | Proposals |
| --- | --- |
| Standard | One to six words with an experience cue or a seen surface form |
| Compound boundaries | Standard spans plus the longest known proper prefix and suffix of each token, with at least three characters |
| Cue-free expansion | All one- to six-word spans within the same punctuation boundaries, plus the compound proposals |

The boundary vocabulary uses only fit ads. The learned model
leaves the current fit ad out of vocabulary and lexical statistics
when constructing each training row. The ranker architecture,
split, calibration method, and target-binding implementation are
shared. Fit each anchor ranker and threshold on its own candidate
distribution. The target binder's outcomes with annotated
anchors are checked to ensure it stayed fixed.

The budget control sends the highest-scoring nonoverlapping
compound-route anchors to the binder, with a per-ad cap from the
number emitted by the standard route. Both select one scope per
anchor. The cap uses the standard model's output count and
observes no test labels.

## Results

| Route | Anchor proposals | Gold anchors in menu / 318 | Anchors sent to binder | Correct top-one links / 561 | Exact pair F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 5,400 | 224 | 263 | 82 | 19.9% |
| Compound boundaries | 5,900 | 267 | 340 | 112 | 24.9% |
| Cue-free expansion | 183,876 | 297 | 339 | 96 | 21.3% |
| Compound boundaries, matched per-ad binder budget | 5,900 | 267 | 263 | 79 | 19.2% |

Compound boundaries add 9.3% to the raw anchor menu and preserve
43 more annotated anchors. With its own operating threshold, the
route sends 77 more anchors to the binder and gains 30 links.
The paired edge-recall gain is +5.35 percentage points, with a
document-grouped 95% interval from +3.29 to +7.75. At the same
per-ad binder budget, the change is −0.53 points (−1.76 to +0.55).
This comparison isolates whether the compound route orders the
highest-value anchors better at the original capacity. Its observed
ranking does not recover the unconstrained gain.

The broad route generates 34 times as many spans as the standard
route. Its exact pair F1 trails the smaller compound menu. When
each route selects a fixed target count on development, the
standard route selects three and issues 789 edges on test. The
compound route selects two, issues 680 edges, and obtains 159 strict
hits versus 156 for standard. The paired recall difference spans
zero. Target fanout and anchor count therefore need separate budgets.

The test uses raw ad text but concerns the annotated GOJA relation
types. The experiment does not measure service latency or expert
relevance. Slurm tasks 46758_0 and 46758_1 on ws3 reproduce the
aggregate metrics and outcome signatures.

## Reproduce

Obtain the pinned anonymized ZIP as in the
[raw-experience benchmark](../goja-raw-experience-scope-v1#reproduce).
Run on a Slurm compute allocation. Repeat with `--reverse`.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.retrieval.goja_anchor_candidate_expansion GOJA_ZIP NEW_RESULTS
```

[Aggregate results](results.json)
