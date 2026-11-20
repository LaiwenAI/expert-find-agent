# Lock a selective role decoder before evaluating the full vocabulary

The [fresh role-search replay](../plos-fresh-role-search-v1) found that
near-role counterfactual training helped in pairwise verification
yet attached more wrong roles than the single-clause head in full
vocabulary search. This test evaluates a decision rule for the full
vocabulary with the same frozen checkpoints, training rows, and author
binding. The previous fresh corpus becomes the development set.
A separate PLOS ONE snapshot supplies the held-out evaluation. The
runner checks DOI, normalized author names, ORCID and exact normalized
activity wording across the two admitted sets.

Compare the old independent per-role cutoffs, the best of 14 full
vocabulary scores, and the same top-one rule with margin abstention.
The latter two accept only one role per request. A top tie abstains.
The margin is the best score minus the second-best score, including
classes absent from the four retrieved papers. Once a role is chosen,
the author index returns only explicit declarations of that role.
If the role has no source in the shortlist, the method abstains.

For each decoder, choose the margin cutoff on development to maximize
focus-supported slots per five recommendation slots, with at most 10%
wrong-role citations among emitted slots and at least 50% query
coverage. Tie-break by lower wrong-role rate, greater coverage, and
then larger margin. If no development cutoff meets the limits the
decoder abstains throughout holdout; its ungated top-one arm is still
reported. Freeze all cutoffs before scoring holdout. This is an
empirical operating point with no guarantee on transferred risk.
A literal baseline accepts one unambiguous exact role name anywhere in
the activity phrase, so it can pick a mentioned but excluded role.

The endpoint requires the designated paper, correct role, and original
owner. Other papers are unjudged for subject relevance, even if they
cite the correct role. Role errors are measurable for all papers.
Panel queries repeat eight activity concepts and each held-out concept
has two new phrasings. These are authored benchmark prompts without
external review. Paired intervals resample focus papers and describe
this fixed phrase inventory; they do not account for uncertainty
from sampling new phrasings.

## Results

Slurm job `43628` ran on one RTX 5090. Development repeats 59 queries
over 12 focus articles in a 36-paper catalog. The independent holdout
has 29 admitted papers and 84 queries from eight focus papers. Their
admission was checked against 524 previous distinct DOIs. The dense
shortlist contained every focus paper on both panels.

The single-clause and near-role heads select dev margins of 1.61195
and 0.50072. They both emit zero wrong roles on development. The
reranker has no feasible margin under both development constraints.
Its gated abstention is recorded as a calibration failure.

| Holdout policy | Requests issued / 84 | Focus supported / 420 | Wrong role / emitted |
| --- | ---: | ---: | ---: |
| Literal role name | 42 | 34 | 83 / 159 |
| Single-clause, old cutoffs | 74 | 110 | 0 / 300 |
| Single-clause, top one | 84 | 115 | 23 / 342 |
| Single-clause, calibrated | 79 | 115 | 0 / 319 |
| Near-role, old cutoffs | 79 | 115 | 14 / 325 |
| Near-role, top one | 84 | 120 | 0 / 338 |
| Near-role, calibrated | 79 | 115 | 0 / 319 |
| Reranker, old cutoffs | 46 | 39 | 105 / 210 |
| Reranker, top one | 84 | 70 | 173 / 378 |
| Reranker, calibrated | 0 | 0 | 0 / 0 |

The calibrated single-clause decoder preserves all 115 focus
hits obtained by its ungated top-one counterpart while removing 23
wrong-role citations. Relative to its original thresholds, it adds
five supported slots (+1.19 points, descriptive paper-cluster interval
+0.57 to +1.71) and issues five more queries. Role errors remain at
zero in both of those arms. The calibrated heads have the same
holdout counts. An ungated near-role arm has five more hits on this
holdout, but accepts 105 wrong-role citations on development
(105/241), against 43/256 for ungated single clause. The new wording
therefore changes which ungated head performs best. Zero observed
errors in a small panel do not establish the underlying error rate
or demonstrate expertise beyond the declarations. Both calibrated
heads cite 204 documents other than the focus DOI, unjudged for fit.

Summing measured query encoding/search and role scoring/selection
times, the two top-one head paths have local P95 of 14.73 and 14.70 ms.
The reranker is 21.35 ms. The margin filter adds no model call.
This is a warmed process with structured declarations already parsed;
the numbers do not include network serving, model loading, or role
extraction from arbitrary text. The single-clause margin policy is
a candidate for declared single-activity requests. Multi-activity,
out-of-vocabulary and no-qualifying-person requests need separately
labelled panels.

[Aggregate results](results.json)

## Reproduce

The fixed holdout DOI and XML hashes appear in [manifest.json](manifest.json).
The capture used PLOS ONE full research entries dated 2025-01-01 through
2026-10-31. It selected up to 16 recent hits per commuting and green
finance title query, and up to 15 per university learning and carbon
economics query. Fetch pinned records from the
[PLOS manuscript XML endpoint](https://api.plos.org/text-and-data-mining.html)
as `pone.<suffix>.xml`. Apply the development manifest from the earlier
study and use its frozen model checkpoints. Run with a Slurm CUDA
allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.role_vocabulary_gate \
    DEV_XML benchmarks/plos-fresh-role-search-v1/manifest.json \
    benchmarks/plos-fresh-role-search-v1/wording.json \
    HOLDOUT_XML benchmarks/plos-role-vocabulary-gate-v1/manifest.json \
    benchmarks/plos-role-vocabulary-gate-v1/wording.json \
    benchmarks/plos-credit-edits-v1/role-training.json \
    benchmarks/plos-factorial-role-training-v1/results.json \
    QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS.json
```
