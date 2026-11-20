# Bound sparse candidates before assigning exact dense ordinals

Use the full-list reciprocal-rank fusion contract of the earlier
[vectorized replay](../vector-hybrid-rrf-v1). Both channels order by
decreasing score and increasing document position on ties. Report
the first 128 fused records in exact order.

Compare three implementations on the same score inputs:

- **Array partial** is the published vectorized baseline. It sorts
  both channels in full, then uses partial selection on fused scores.
- **Packed full** uses a single unsigned integer key for each finite
  float32 dense score and its document position. It retains full dense
  sorting. Normalize signed zero to match the reference tie rule.
- **Bounded** selects and sorts only the first 512 dense positions.
  It computes their exact fused scores using the full sparse list.
  The 128th best score among those positions is a lower bound on the
  final cutoff. A document outside the dense prefix can contribute
  at most `1 / (60 + 513)` from dense retrieval. Inactive sparse
  documents outside the prefix cannot reach the cutoff, because the
  prefix already includes its first 128 dense positions. An active
  sparse document proceeds only if its reciprocal sparse rank plus
  the dense upper bound can reach the cutoff. Equality is retained
  so that final document-position ties stay exact.

For the retained outside positions, sort their packed keys. Use
insertion buckets to count every eligible dense key less than or
equal to each retained key. Their cumulative bucket counts are
exact one-based dense ranks because the packed keys are unique.
Calculate exact fusion scores only for the prefix and the retained
outside positions, then apply the original tie order. This applies
monotone score bounds to the full-list RRF contract with resident
score arrays. It uses the same candidate universe as [reciprocal-rank
fusion](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf).
The bound follows the broader approach of [threshold
aggregation](https://www.sciencedirect.com/science/article/pii/S0022000003000266).

The timed large-catalog component starts after dense scores are
available and includes BM25, rank operations and final selection.
Dense score computation still covers all eligible documents. The
452-article publisher replay separately measures the complete
warm request from parsing through expert and citation serialization.
Its candidate lists are shorter than 512, so the bounded route
uses all dense positions there. The 1,500 generated equivalence
checks cover signed zeros, tied scores, excluded documents, and
shuffled sparse row IDs.

## Results on RTX 5090

Both tasks of Slurm array `44650` completed on ws5. All 246 LitSearch
queries retain the same ordered top 128. On the mixed publisher catalog
all 211 queries retain their original ranked lists and exact serialized
responses. These responses include the same 85 PLOS and 46 eLife known
matches as the earlier mixed-catalog replay.

| Scope | Array partial P95 | Packed full P95 | Bounded P95 |
| --- | ---: | ---: | ---: |
| 64,183 LitSearch articles, ranking component | 7.71–7.83 ms | 5.44–5.53 ms | 5.03–5.07 ms |
| 452 articles, ranking component | 0.154–0.156 ms | 0.165 ms | 0.170 ms |
| 452 articles, complete warm request | 10.49–10.60 ms | 10.59–10.70 ms | 10.47–10.57 ms |

The median LitSearch query activates 56,820 sparse positions. Beyond
the sorted dense prefix of 512, the bound retains a median of 68
positions for exact dense ordinal calculation, with a P95 of 144 and
a maximum of 202. Packed full sorting accounts for most of the
improvement against the prior array implementation. The bound
saves another 0.40–0.45 ms against that stronger control on the
measured component. On the small catalog the original vectorized
implementation remains the simpler and faster ranking component,
and complete request P95 is similar.

## Reproduce

Use the frozen PLOS/eLife snapshots, model, and LitSearch projection
and dense cache from the [vectorized benchmark](../vector-hybrid-rrf-v1).
Run twice on a Slurm compute node with a GPU, fresh output paths,
and `--order-offset 1` for the second run.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.bounded_dense_rrf \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json \
  LITSEARCH_PROJECTION LITSEARCH_VECTORS NEW_RESULTS
```

[Replicated timings and exactness checks](results.json).
