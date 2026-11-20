# Distinguish a qualified author, no match, and unresolved interpretation

Build two-activity requests from the published
[full-vocabulary holdout](../plos-role-vocabulary-gate-v1). This is a
new conjunction task on reused papers and activity phrases. The request
refers to a specific study identified by its topic clue. Both activities
are present somewhere in the focus paper. Positive cases have at least
one coauthor credited for both. Matched negative cases have each
activity credited to different people, with nobody credited for both.
Keep up to three papers per condition for each activity pair, balancing
the two conditions by pair. Evaluate both existing phrasings. Clause
boundaries and the conjunction are given to all routes.

Each route searches actual titles and abstracts, takes the top article,
and uses the same publisher-declared author index. An incorrect top
article cannot satisfy a request. A negative decision is correct only
when the correct paper is retrieved and no author qualifies. A label
missing from this paper says nothing about work elsewhere. Compare
these routes with the same underlying activity scorer where applicable:

- The known-role control isolates source retrieval and author binding.
- Article union detects that the two predicted roles appear somewhere
  in the article and broadcasts them to all coauthors. This diagnostic
  exposes attribution leakage.
- Point prediction intersects the author postings for the two best
  distinct labels. The strict option abstains if either margin fails
  the cutoff frozen in the single-role experiment.
- The ambiguity-aware options use the 80th percentile and maximum of
  gold role score deficits on eight separate development phrases.
  Each allowed role pair must contain two distinct roles. Return
  common authors if present in every interpretation, return no match
  if every admitted interpretation is empty, and leave disagreements
  unresolved.
- The frozen reranker batch compares both clauses to all 14 labels and
  applies the same point and author-intersection rules.

The ambiguity-aware rule generalizes the prior
[common-evidence check](../plos-robust-role-bundles-v1) by separating
ambiguous no-match decisions from unresolved results and by including
live retrieval. Its guarantee is conditional on correct source
retrieval, complete publisher declarations, and inclusion of the
intended interpretation. The development quantiles do not guarantee
the last condition.

An exact result retrieves the correct paper and fully reproduces its
qualifying author set, including the empty set. A supported recommendation
may return a nonempty proper subset. Report incorrect recommendations
and incorrect no-match decisions separately. Paired intervals resample
whole focus papers. P95 sums measured query encoding and topic search,
model forwards, and route logic over three timing repeats. It excludes
model loading, JATS parsing, persistent I/O and service queueing.

## Results

Slurm job `43634` ran on one RTX 5090. The 128 queries cover eight
reused focus papers and 23 role pairs, with 64 queries in each
qualification stratum. All target papers rank first.

| Route | Exact / 128 | Correct no match / 64 | Any wrong recommendation | Wrong no match | Unresolved | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Known roles | 128 | 64 | 0 | 0 | 0 | 7.68 ms |
| Article union | 0 | 0 | 124 | 2 | 2 | 16.02 ms |
| Point prediction | 119 | 61 | 2 | 5 | 2 | 16.00 ms |
| Strict single-role margin | 110 | 55 | 0 | 0 | 18 | 16.00 ms |
| Role sets, 80th percentile | 119 | 61 | 2 | 5 | 2 | 16.02 ms |
| Role sets, development maximum | 121 | 62 | 0 | 0 | 7 | 16.01 ms |
| Batched reranker | 81 | 47 | 3 | 14 | 30 | 30.86 ms |

The maximum deficit of 1.60094 admits the intended interpretation on
all 128 queries. The 80th percentile is zero and reproduces point
predictions on this panel. Among the 64 requests with no qualified
author, article union wrongly recommends on 63. Point prediction
reduces that number to two. The wider ambiguity-aware route has no
observed wrong recommendations or false no-match decisions, and
resolves 11 more cases than the strict single-role-margin control at
similar latency. Against point prediction, its exact-response gain
is +1.56 percentage points (descriptive paper-cluster interval
0.00 to +3.00). Eight papers and repeated phrases limit the inference.

The result supports checking ambiguity against available evidence
before abstaining on all close scores, within the specific-paper and
complete-declaration setting. Queries that allow arbitrary sources,
contain unsegmented clauses, or ask for roles outside the vocabulary
still require separate evaluation. [Aggregate results](results.json)

## Reproduce

Use the held-out JATS files and frozen manifests and wording from the
full-vocabulary experiment. Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.conjunctive_role_gate \
    benchmarks/plos-fresh-role-search-v1/manifest.json \
    benchmarks/plos-fresh-role-search-v1/wording.json HOLDOUT_XML \
    benchmarks/plos-role-vocabulary-gate-v1/manifest.json \
    benchmarks/plos-role-vocabulary-gate-v1/wording.json \
    benchmarks/plos-credit-edits-v1/role-training.json \
    benchmarks/plos-factorial-role-training-v1/results.json \
    benchmarks/plos-role-vocabulary-gate-v1/results.json \
    QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS.json
```
