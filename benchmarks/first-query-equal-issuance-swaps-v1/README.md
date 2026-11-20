# Compare rescue evidence at equal issued recommendation counts

Use the eLife and PLOS catalogs, topic splits, both declaration masks,
two-role requests, and pinned BGE-small and MiniLM checkpoints from
[the previous headroom test](../first-query-routing-headroom-v1).
The topical ranking, first four source reads, scores, and first-stage
recommendations are identical between arms. Both arms verify two
additional sources. Local rescue prioritizes role-compatible unseen
sources within the original topical top eight. Global rescue takes
those sources in topical order across the catalog. If a route lacks
two compatible choices, fill its verifier batch with unseen sources
from the original pool as in the existing matched-read control.

Keep the final same-person ranking and original topical threshold
fixed. Compare ordered witness lists at two equal output counts:

1. The number of verified people already supported by the common four
   initial reads. This ceiling is known before either rescue executes.
2. The smaller final issuance of the arms. This is a paired
   retrospective check and requires both outcomes.

A witness is a deduplicated person and the DOI of their supporting
article. Count a source replacement for an incumbent separately from
a new person. Report added and removed publisher-tagged witnesses and
topic-paired intervals over the same fixed five slots. Topic tags are
incomplete and cannot adjudicate evidence sufficiency.

With optional local outputs, create a blinded case list for changed
witnesses and a separate sealed direction key. Cases expose the public
source DOI, article title and abstract, author position, claimed roles
and requested topic. They do not include the arm or topic tag label.
Deduplicate case cards across masks and caps using query, source and
author position. The packet supports independent topic and role
assessment. Different sources for the same person can still appear in
different cards.

## Results

There are 106 distinct requests across the two publishers. Each
request is repeated at two declaration masks. All eLife arms and
the sparse PLOS mask give identical matched recommendation lists
under either issuance cap. The full PLOS mask gives:

| Equal issuance cap | Total slots per arm | Queries with a changed list | New people | Tagged additions / removals |
| --- | ---: | ---: | ---: | ---: |
| Supported people after the initial four reads | 50 | 1 | 1 | 0 / 0 |
| Smaller final issuance of the two arms | 63 | 2 | 2 | 1 / 1 |

The paired tag-backed P@5 difference is zero in both caps and both
CPU replays match. Four unique evidence cards cover the three observed
contrasts, with overlap across the output caps. Independent judgments
are pending. In this action and ranking setup, widening rescue scope
rarely replaces any incumbent and gives no net tag-backed replacement
gain. More targeted retrieval or a different ranking objective is
required to test better replacements. The experiment does not
adjudicate topic or personal contribution claims.

[Aggregate results](results.json)

## Reproduce

Restore the same verified projections and checkpoints as in the
[frozen routing study](../first-query-topic-role-router-v1).

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.role_first_query_swaps ELIFE_PROJECTION PLOS_PROJECTION \
  benchmarks/plos-topic-role-holdout-v1/manifest.json \
  benchmarks/elife-first-query-factorial-v1/results.json \
  BGE_MODEL_DIRECTORY FLASHRANK_CACHE NEW_RESULTS
```

To generate blinded cards, pass `--review-packet` and `--private-key`
with distinct fresh local output paths. Give only the cards to an
independent reviewer. They should judge topical support and whether
the article actually assigns both roles to the author at the specified
position, explaining each unresolved case. Keep the direction key
separate until the review is recorded.
