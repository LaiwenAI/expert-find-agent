# Confirm a source against an alternative and abstain when neither matches

Evaluate source confirmation on 246 author-written [LitSearch](https://aclanthology.org/2024.emnlp-main.840/)
questions after the controlled PLOS contributor experiment. Papers
have relevance references. They have no person-level attribution
labels in this benchmark. This is a source-stage transfer test.

Search the complete, pinned 64,183-paper title-and-abstract corpus
with the existing exact dense retrieval implementation and frozen
Qwen3-Embedding-0.6B. Every route sees the same two candidates from
the dense ranking. Withhold all released target papers before
retrieval in the second state of each query. The test questions
were previously used for source confirmation. Threshold calibration
uses 96 separate inline questions without shared target IDs or
exact normalized question strings.

| Method | Model decision | Evidence presented |
| --- | --- | --- |
| Reranker | Independent yes/no, Qwen3-Reranker-0.6B | Complete title and abstract |
| Binary | Independent yes/no, Qwen3-4B-Instruct-2507 | First 160 words of each title and abstract |
| Choice full | First, second, or neither, 4B | Same content as Binary |
| Choice short | First, second, or neither, 4B | Title and up to 64 words from two query-aligned abstract sentences |
| Choice balanced | Average decisions for both candidate orders, 4B | Same content as Choice short |

The comparison methods use one 4B forward per request. The balanced
method runs two prompt rows in that forward. The best paper's
score is its logit margin over the neither label. The gap is
the difference between the two paper margins. Use the same
absolute-score and gap threshold search for all arms. Run one
fixed calibration procedure at 2%, 5%, and 10% ceilings on
unjudged returns separately for present and withheld views.
Aggregate paired intervals use target-connected groups.

Rank the two short-window sentences by the sum of query-term IDF.
Keep selected sentences in original order. Derive IDF from the
document inventory alone. The prompt presents one candidate
order chosen by query hash. The balanced arm presents both and
averages each candidate's margin after mapping positions back to
the source. The recorded order-flip count compares the two
unthresholded paper preferences, independent of later abstention.

A returned paper outside the released references is unjudged.
Withholding measures substitution against intended references.
It cannot determine whether the alternative is relevant to the
question. These empirical calibration ceilings constrain the
calibration returns. They provide no test-time safety guarantee.

Request timing includes live query encoding, exact retrieval,
short-window selection when used, and the actual model forward.
It excludes index construction, model loading, and service queues.

## Results

Slurm job 43266 completed on one RTX 5090 on ws5. The two
dense candidates contain a released target on 138 of 246
author-written questions. All arms receive the same pair
of papers in every view. The reranker reproduces the prior
reference-return counts at all three ceilings. Its native
token cap truncates two rows in the present view and three
in the withheld view.

Each cell shows **known return / unjudged return with targets
present / unjudged return after withholding**.

| Method | 2% calibration ceiling | 5% calibration ceiling | 10% calibration ceiling | Present P95, ms |
| --- | ---: | ---: | ---: | ---: |
| Reranker | 23 / 2 / 3 | 30 / 7 / 9 | 42 / 8 / 14 | 25.22 |
| Binary | 12 / 2 / 2 | 14 / 2 / 4 | 35 / 5 / 9 | 46.68 |
| Choice full | 20 / 1 / 3 | 34 / 0 / 2 | 46 / 4 / 8 | 40.74 |
| Choice short | 11 / 0 / 0 | 15 / 2 / 5 | 47 / 5 / 11 | 30.39 |
| Choice balanced | 14 / 1 / 4 | 31 / 2 / 6 | 53 / 9 / 14 | 44.94 |

At the 5% calibration ceiling the full choice prompt returns
34 known papers compared with 14 for binary scoring at the
same checkpoint and candidate depth. The paired increase is
+8.13 percentage points (95% target-connected interval +3.66
to +12.65). It lowers P95 from 46.68 to 40.74 ms. Against
the faster 0.6B reranker it returns four more known papers
and seven fewer unjudged papers after withholding, with
15.53 ms more P95 latency. This comparison reports observed
operating points on the reused test panel.

Shortening the choice prompt from 486 to 273 mean input
tokens lowers P95 to 30.39 ms. At the 5% ceiling it returns
19 fewer known papers than the full choice prompt. At 10%
it returns one more known paper and three more unjudged
papers after withholding. Reversing candidate order changes
the preferred paper before abstention in 36 present and 72
withheld queries. Averaging the two orders increases P95
to 44.94 ms and has different trade-offs across ceilings.

The full choice route is a higher-coverage alternative to
binary scoring at the same 4B size. The small reranker remains
the latency reference. The results isolate source selection
and do not adjudicate the substituted papers or personal roles.

[Results and paired intervals](results.json)

## Reproduce

Use the previously pinned LitSearch projection, dense cache and
local Qwen checkpoints. Run on Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.litsearch.contrastive_source_confirmation \
  PROJECTION DENSE_CACHE EMBEDDING RERANKER QWEN4B NEW_RESULTS
```
