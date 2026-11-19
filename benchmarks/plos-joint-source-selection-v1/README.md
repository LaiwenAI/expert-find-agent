# Use author-specific role evidence when selecting sources

Reuse the [55-paper independent PLOS snapshot](../plos-source-gate-transfer-v1/README.md). It was constructed by excluding previous publications and contributor identities. Generate three requested duties across two sources using the [linked-witness pair generator](../../experiments/credit/linked_witness_dispatch.py). Each matched edit changes the role or which publication must substantiate it. Each duty uses one of two phrase inventories, publisher-independent NISO examples or the existing external boundary wording. The same small phrase banks recur across cases. Source clues use either the last two abstract sentences or rare keywords from those sentences. Each request is evaluated with both sources available and with each intended source individually removed.

Freeze the embedding checkpoint, ridge head training phrases, role cutoff, lexical and dense retrieval, and reciprocal-rank fusion. Form ordered source pairs by shared author identity, with each side within its top 12 and at least one side within its top four. Rank pairs by sum of source ranks, then worst rank and stable identifiers. Every returned expert must pass the same author-specific membership gate on the indexed publisher declarations for both sources.

| Policy | Pair selection |
| --- | --- |
| Source top four | Check only the first linked pair from the top four of both source lists. |
| Source first | Check only the first pair from the wider list. |
| Source backfill | Check up to the first two pairs and return the first author-verified recommendation. |
| Pooled priority | Scan until each duty is supported by at least one linked author, then require a common author to return a recommendation. |
| Owner priority | Scan until a single linked author passes every requested duty. |

The last two policies have the same maximum eligible pool. They may stop after different amounts of work. All methods share one online forward pass for clues and role phrases per request. Pair inspections access indexed publisher declarations. No full texts or per-pair model calls are needed. Timing includes shared encoding, ranking and timed membership checks. It excludes index construction, storage and service transport. Count a recommendation as supported only when it cites the intended sources and an author whose actual declarations satisfy every requested duty. The test assesses selection of intended sources, and alternative sources are errors under that contract.

## RTX 5090 results

Slurm job 42556 tests 40 positive and 40 matched unsupported requests drawn from four linked publication pairs. It crosses two wording banks, two clue types and three source-availability conditions. The 640 masked evaluations reuse the same requests. They do not represent independent risk trials.

| Both sources present | First wider pair, supported / unsupported | Two-pair backfill, supported / unsupported | Author-guided full scan, supported / unsupported |
| --- | ---: | ---: | ---: |
| NISO wording and result clues | 28 / 7 | 28 / 12 | 28 / 12 |
| External wording and result clues | 25 / 3 | 25 / 7 | 25 / 7 |
| NISO wording and keywords | 15 / 2 | 21 / 6 | 21 / 6 |
| External wording and keywords | 14 / 3 | 19 / 4 | 19 / 4 |

Each cell uses 80 evaluations, 40 of which can yield a published-role-supported match. The top-four and first-wide policies agree on these both-present rows. With keywords, two-pair backfill recovers six additional supported queries for the NISO wording and five for the external wording. It adds four and one unsupported recommendations respectively. The added unsupported recommendations in both keyword rows cite the intended papers but misread the author's duties. With result clues, backfill adds only wrong-source recommendations. The top-four, first-wide and two-pair routes issue 0, 3 and 5 unsupported recommendations across 640 masked evaluations.

The full owner-priority scan and pooled priority match the two-pair backfill counts. The owner-priority scan inspects 18 and 24 additional pairs in the two result-clue rows with no additional recommendation. Factored P95 stays between 8.93 and 9.10 ms in the both-present rows. The wider pool includes the intended pair for just 30 of 40 positive keyword requests in each wording condition. With the intended pair supplied, the frozen role gate itself confirms only 28 of 40 NISO positives and 25 of 40 external positives. Both source association and role interpretation therefore limit further gains. Retain two-pair backfill as the inexpensive comparison for short clues and explicitly score its errors. This panel does not establish an expert-quality or completeness improvement from author-guided expansion. [Aggregate results](results.json)

## Reproduce

Rebuild the independent projection and panel using their [capture and preparation manifests](../plos-source-gate-transfer-v1/README.md#reproduce). Use the [NISO inventory and ridge training inputs](../plos-shared-embedding-witness-v1/README.md#reproduce), the [boundary phrase inventory](../plos-credit-external-v1/boundary-phrases.json) and the pinned [ridge calibration aggregate](../plos-shared-embedding-witness-v1/results.json).

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.joint_source_selection \
  HOLDOUT_PROJECTION HOLDOUT_PANEL NISO_EXAMPLES BOUNDARY_PHRASES ROLE_TRAINING \
  RIDGE_CALIBRATION_RESULTS QWEN_EMBEDDING NEW_RESULTS
```

Run in a Slurm GPU allocation. The aggregate pins the script, inputs and checkpoint.
