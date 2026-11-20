# Test whether selective title lookup transfers across CoF collections

The [SciRepEval dual-view pilot](../real-title-dual-view-v1) found
that always reserving the second of two source slots for a title-only
result worsened known-positive coverage. Test whether a selective
policy can preserve the two-source budget and identify the requests
where the title view should replace the second title-abstract source.

Use release collections NIPS, KDD, and SciRepEval. The SIGIR
release has no title fields populated for its 1,740 profile sources,
so it cannot support this comparison. Exclude queries with IDs, normalized titles, or normalized full text present
in their source collection. Remove normalized query titles shared
across collections before fitting or evaluation. Fit thresholds and
weights on two collections and evaluate on the remaining collection.
SciRepEval motivated the experiment; its fold is an exploratory check.
NIPS and KDD test transfer within the CoF benchmark. Once the primary
run showed that only two NIPS queries survived, run a sensitivity
analysis that preserves the full test collection and removes matching
titles from the training collections for each fold. The sensitivity
folds share some test queries and should be read separately.

Keep the strongest title-abstract source. The second slot is the next
title-abstract source unless the route fires, in which case it uses
the highest scoring distinct title-only source. A query can be
routed using four features computed before title lookup: the best
title-abstract score, the first-to-second gap, the second-to-eighth
gap, and log abstract length. Compare these policies:

- Never replace and always replace.
- Replace below the fitting collections' first-to-second gap quartile.
- Replace when a ridge model predicts a gain above zero and above
  the fitting collections' 75th percentile. The target is the mean
  of the soft and hard known-positive coverage changes. The penalty
  is 0.25 times the number of training queries on standardized features.
- Compare each policy's net change with the expected net change from
  uniformly random replacement at the same number of title scans.

Use original sparse reviewer labels. Report the number of held-out
queries reaching a known positive, gains and losses relative to no
replacement, judged and returned pairs, and paired query-bootstrap
intervals. This is a counterfactual policy replay. The title-scan
count estimates additional retrieval work. Neither wall time nor
enterprise recommendation and attribution quality is measured here.

The title-abstract source vectors and query vectors come from the
existing CoF cache. Verify input order and text digest, matrix hash,
checkpoint provenance, and the historical last-token encoder
digest. That encoder predates the optional mean-pooling mode.
Use its current default last-token path and the same checkpoint
for encoding source titles. The cached source and query batching
differs from the live SciRepEval pilot, so report its controls from
the current replay.

## Results

Slurm job `43997` on ws5 reproduced all primary aggregates from
job `43989` before reporting the sensitivity analysis.

| Primary held-out fold | Queries | Title-abstract positives | Always replace | Low-gap rule | Learned gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| KDD | 170 | 28 / 11 | 35 / 15 | 29 / 11 | 29 / 12 |
| SciRepEval | 59 | 33 / 18 | 29 / 19 | 32 / 18 | 33 / 18 |
| NIPS | 2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Each result is soft / hard known-positive query coverage.
On KDD, always replacing added 7 soft queries, with a paired 95%
interval of +0.6 to +8.2 percentage points. In SciRepEval it lost
4 soft queries. Thresholds fit on the other collections triggered
75 low-gap scans and 43 learned scans on KDD. The learned gate
retained 1 of the 7 net soft gains. Uniform replacement of 43
queries had an expected 1.77 net soft gains in that fold.
Of the 34 eligible NIPS queries, 32 share a normalized title with
SciRepEval. The two globally disjoint queries provide little
evidence about NIPS.

With overlapping titles removed from training only, NIPS recovered
34 test queries. Always replacing found one extra soft positive
query; the learned gate scanned twice and gained none. In KDD, the
learned gate scanned 34 times and had zero net gain on both rubrics.
The learned gate also had zero net gain in SciRepEval. The low-gap
rule realized two extra soft positives across 57 KDD scans, below the
2.35 expected from the random policy at that scan count.

A fixed title slot can help within some collections. Simple
pre-lookup features did not reliably identify when it helps across
collections. Keep title-abstract retrieval as the default and treat
title supplementation as a mechanism to test where specific
evidence gaps provide an actionable signal.

[Aggregate replay](results.json)

## Reproduce

Prepare the existing CoF Torch embedding caches with the historical
last-token encoder before the mean-pooling change. Use the pinned
CoF archive, a fresh output file, and a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.cof_title_route_transfer QWEN_EMBEDDING NEW_RESULTS
```
