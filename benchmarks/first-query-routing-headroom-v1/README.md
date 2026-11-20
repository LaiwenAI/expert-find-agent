# Bound how much a nomination router could improve the current actions

Use the pinned eLife and PLOS projection files, topic splits, two
declaration masks, encoder, reranker and verifier threshold from the
[pre-verification routing study](../first-query-topic-role-router-v1).
Run the same fixed-six-reading policies used by the
[matched-read study](../elife-first-query-equal-reads-v1). The local
action set contains topic-first with local rescue and person-first
with local rescue. The expanded action set also contains person-first
with global rescue. Source ordering and score batches are pinned.
Only an identical query and ordered source batch can reuse scores.

Treat each action's evaluated output as known after the fact. Select
whole outputs for each request. Report maximum tag-backed slots under:

- Unrestricted choice among the available actions.
- No more off-label slots on each request than in topic-first local.
- No more issued slots on each request than in topic-first local.
- No more total off-label slots across a panel than in topic-first local.

For the panel-wide constraint, use exact dynamic programming indexed
by total off-label slots. Each request adds one action to the frontier.
At each cost retain the largest attainable tag-backed count. This
allows a reduction on one request to offset an increase on another.
Keep those results distinct from the stricter per-request constraint.
Every choice uses exactly six verifier reads. Timing and the costs
of learning or executing a router are outside this experiment.

These are hindsight bounds on observed publisher tag labels and
fixed action outputs. Topic tags are incomplete. The bounds use
test labels and cannot be deployed as a selector. Their purpose is
to separate a limited action set from a failed routing model.

## Results

The table shows tag-backed slots. The local and expanded bounds
enforce the topical route's per-request off-label count. The last
column enforces the topical route's per-request issuance count
and uses the expanded action set.

| Test panel | Topic-first local | Best local choice with hindsight | Best choice including global with hindsight | Best choice without extra issuance |
| --- | ---: | ---: | ---: | ---: |
| eLife headings, each mask (24 requests) | 5 | 5 | 5 | 5 |
| eLife keywords, sparse (36) | 49 | 49 | 51 | 49 |
| eLife keywords, full (36) | 49 | 51 | 52 | 49 |
| PLOS leaf topics, sparse (46) | 48 | 53 | 54 | 48 |
| PLOS leaf topics, full (46) | 51 | 55 | 56 | 51 |

Allowing the off-label count to shift between requests gives the
same tag-backed maxima in these cells. In all cases, an output can
gain tag-backed slots only by emitting additional recommendations.
There are no observed gains for any request when selecting among
whole outputs with the same or fewer recommendations. This does
not test trimming lists or replacing individual recommendations.
The strongest available evidence for this action set is therefore
additional tag-backed coverage, with little evidence on ranking at
matched issuance. Better routing alone cannot improve that endpoint
within these three frozen whole-output actions.

The limits leave room to improve the source-backed ranking action
itself, for example by verifying a replacement for a specific
candidate at a fixed number of issued seats. That experiment would
need independent judgments of the incoming and displaced evidence.
Two CPU replays give identical reports. A separate synthetic check
compared the pooled optimization with exhaustive enumeration on
300 randomized examples and matched every result.

[Aggregate results](results.json)

## Reproduce

Restore the projection and model inputs from the
[frozen routing benchmark](../first-query-topic-role-router-v1).

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.role_first_query_headroom ELIFE_PROJECTION PLOS_PROJECTION \
  benchmarks/plos-topic-role-holdout-v1/manifest.json \
  benchmarks/elife-first-query-factorial-v1/results.json \
  BGE_MODEL_DIRECTORY FLASHRANK_CACHE NEW_RESULTS
```
