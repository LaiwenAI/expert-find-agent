# Spend the rescue budget on candidates with distinct role evidence

Use the same 106 distinct eLife and PLOS two-role requests, two
declaration masks, BGE-small hybrid ranking, and MiniLM L12 verifier
as the [equal-issuance study](../first-query-equal-issuance-swaps-v1).
All arms share the first four topical reads and read exactly two
additional sources. Compare local rescue, ordinary global rescue,
and a global rescue that prioritizes new people. Final ranking and
relevance threshold are the same in every arm.

The new-person route starts from the accepted first-stage identities.
Among unread sources in the first 16 topical ranks, prioritize
visible declarations binding both requested roles to at least one
identity absent from that set. Keep topic order within this filter.
After selecting a source, treat its known identities as covered
when choosing the second source. Fill any remaining rescue slots
from the ordinary global compatible order. Previously unindexed
declarations remain eligible through this fallback. The top-16
bound is fixed and uses no judgments from either test panel.

As before, hold issuance at the shared first-stage count, and
separately at the smaller final count in each comparison.
Report person replacement, witness replacement, gains and losses
under publisher tags, and paired intervals grouped by topic.
The first cap is computable without seeing the other arm.
The second is only an evaluation diagnostic. The paper evidence
is source- and role-bound. Topical adequacy still requires
independent assessment, particularly for off-label citations.

## Results

The ordinary global control reproduces the previous study in every
cell. At the common first-stage output cap, the new-person route gives:

| Panel | Equal slots per route | New people vs local | Tagged additions / removals |
| --- | ---: | ---: | ---: |
| eLife keywords, sparse | 54 | 6 | 5 / 6 |
| eLife keywords, full | 54 | 0 | 0 / 0 |
| PLOS, sparse | 50 | 8 | 6 / 6 |
| PLOS, full | 50 | 1 | 0 / 0 |

eLife heading requests have no swaps. At the smaller-final-list cap,
the PLOS sparse panel gains one tag-backed slot (paired interval
−0.87 to +1.74 P@5 points). The eLife keyword sparse panel loses two
(−2.78 to 0.00 points). The primary count-based comparison shows no
gain. The selector can change who gets recommended at equal effort.
The current topical verifier and ranking do not consistently turn
those replacements into better tagged recommendations.

Two replays yield byte-identical aggregate reports. Thirty-nine
local, blinded evidence cards cover 28 changed contrasts across
masks and caps; their independent review is pending. The reports
contain aggregate counts only.

[Aggregate results](results.json)

## Reproduce

Restore the verified eLife and PLOS projections and open checkpoints
used by the previous equal-issuance experiment.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.role_first_query_person_rescue ELIFE_PROJECTION PLOS_PROJECTION \
  benchmarks/plos-topic-role-holdout-v1/manifest.json \
  benchmarks/elife-first-query-factorial-v1/results.json \
  BGE_MODEL_DIRECTORY FLASHRANK_CACHE NEW_RESULTS
```

Use both `--review-packet` and `--private-key` with fresh local paths
to reproduce the blinded cards and their separate direction key.
