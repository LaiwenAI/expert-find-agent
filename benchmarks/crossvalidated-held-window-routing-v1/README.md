# Transfer single-window routing to later community questions

The [window-factorial pilot](../crossvalidated-witness-window-v1)
found five gains and five losses when moving the verifier's 850-character
answer window. Fit one cutoff on the 2024 development cohort. Evaluate
on 2025 Cross Validated duplicate questions captured with the same
earlier answer-content cutoff. Exclude any held-out query that shares a
known answer ID or canonical question ID with development, or has an
80% token-set title overlap with any development query.
Both catalogs also contain historical distractors. Shared distractors
do not provide training labels.

Freeze the dense author shortlist and same-author additional checks.
The 0.6B verifier scores the same two ordered batches for every arm.
The route chooses only one view per answer. Score the fractional
increase in IDF-weighted request term coverage between lead and
targeted windows. Choose the targeted view only above a single
development cutoff. Optimize observed exact-witness hits on
development, breaking ties by fewer switched views and then a
larger threshold. The fitting code reuses the two pre-scored views
to evaluate candidate thresholds. A live development replay
separately measures the chosen mixed batch. The held-out run
executes the route with the frozen cutoff and no label access
in candidate or window selection.

Keep lead-only and targeted-only as equal-character and equal-check
controls. The request-length and prompt diagnostic holds the lead
window fixed and crosses the original 950 request characters versus
up to 1,800 with the original and explicitly answer-focused
instruction. Report any inputs capped by the model's packed token
limit and the paired result on longer requests that remain uncapped
in both compared arms. Longer requests can increase compute even
when the answer window stays the same length.

P95 includes view selection, two verifier batches, and final citation
selection. Dense retrieval is fixed outside the timed stage. Known
targets come from the community-linked accepted answers. Results
measure their recovery in the bounded source catalogs. The absence
of a labeled witness for other candidates does not provide negative
relevance judgments.

## Results

Slurm job `43714` ran on one RTX 5090. Four of the 31 captured
questions share a canonical source with development. The retained
27 questions use a 323-answer catalog and 206 identical checks per arm.
All methods see the same 26 queries with a proposed known witness.

The development route selects a normalized IDF gain cutoff of
0.06054. It switches 65 of 526 answer views and retains 42
known-witness queries, versus 38 for lead. Live development
decisions reproduce these counts. The route switches 28
out of 206 views on holdout.

| Verification setup | Known witness queries / 27 | Cited answers | Capped checks | Local P95 |
| --- | ---: | ---: | ---: | ---: |
| Original lead | 12 | 16 | 0 | 62.67 ms |
| Always targeted | 13 | 20 | 0 | 68.15 ms |
| Single-window routing | 12 | 17 | 0 | 65.49 ms |
| Longer request | 13 | 17 | 6 | 75.56 ms |
| Answer-focused prompt | 17 | 27 | 0 | 64.53 ms |
| Longer request and answer-focused prompt | 17 | 26 | 7 | 76.76 ms |

The route's development gain does not transfer. Always-targeted
gains one query at modest local overhead. Extending the request
changes the text for ten queries and gains one known witness over
original lead. One extended-request query has capped inputs.
Among the nine uncapped changes, the net is one extra hit.

The answer-focused prompt alone gains five exact-witness queries
without losing an original hit (+18.52 points, descriptive
shared-answer-cluster 95% interval +3.85 to +34.62).
It also issues 11 more citations. Other cited answers are
unjudged, so a precision or equal-issuance improvement cannot
be inferred. This result makes the verification instruction
the next factor to test under a matched issuance budget. It does
not support wider input windows or lexical window routing as defaults.
The one-community holdout and small observed counts limit generality.

[Aggregate results](results.json)

## Reproduce

Use the frozen 2024 snapshot and
[the 2025 query and answer manifest](holdout-manifest.json).
The latter used nine pages of closed 2025 questions. The API
returned data on the first four pages. The capture verifies
the original-question link and accepted answer and keeps only
text created and last edited before 2023. Exact archive
reproduction requires the cached API receipts. Later public
API requests may reflect updates. Run inference with the same
pinned open checkpoints in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. python -m experiments.community.crossvalidated_witness \
  HELD_SNAPSHOT_DIR --year 2025 --duplicate-pages 9
PYTHONPATH=src:. PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.community.held_window_routing \
    DEV_DIR/snapshot.json DEV_DIR/manifest.json \
    HELD_SNAPSHOT_DIR/snapshot.json HELD_SNAPSHOT_DIR/manifest.json \
    benchmarks/crossvalidated-witness-window-v1/results.json \
    QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS.json
```
