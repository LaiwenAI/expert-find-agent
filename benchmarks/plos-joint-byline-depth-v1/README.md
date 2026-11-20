# Widen topic retrieval while keeping three declaration reads

Reuse the 160 requests, 130 publisher records, publisher-supplied ORCID
bylines and frozen Qwen3 embeddings from the
[open topic-and-role benchmark](../plos-open-topic-role-bundles-v1).
Truncate each topic ranking at 8, 16, 32 or 64. Preserve the same
scoring, declaration checker, read cap and answer rule at every
depth. Topic and exact author-role labels are used only for
evaluation, except that the separate role-union control is unchanged.
A complete reference bundle is reachable when each facet's pool has
one compatible document from the same ORCID, on distinct papers.

## Exact source nomination

The reference scans all ordered pairs across the two rankings.
Its primary pair minimizes the worse topical rank, then the sum
of ranks and then the IDs. For the third source it counts directed
compatible pairs from that source to either of the primary sources.

The owner-postings implementation gathers the first two article
IDs per owner and condition. If one owner's top two lists begin
with different sources, those first sources give its best pair.
If they begin with the same source, the optimum must keep one
first source and replace the other with its second source.
Taking the best pair across owners therefore preserves the
reference ranking. For the third source, count directed pairs
by checking its byline against the two selected sources only.
Shared owners in a document pair are counted once, as in the
reference. Unjoinable candidates retain the topical fallback.
Each timed benchmark call compares the full ordered triple from
both implementations. Also checked 1,000 seeded configurations
with variable overlap, ranking lengths and empty bylines.

## Results

Both Slurm tasks in array `44429` ran on ws5. Frozen embeddings
were scored on RTX 5090. Quality aggregates reproduce in both tasks,
including every metric at depth eight from the previous benchmark.
Timing uses warm CPU planning calls with 10 rotated repetitions on
the 160 requests. It excludes topic encoding, title/abstract ranking,
I/O, publisher-role verification and output assembly. All hybrid
policies below read three declarations across every query.

| Ranked articles per topic | Queries with full reference proof in pool / 160 | Known under pooled topic plan | Known under owner-gap rescue | Joint plan known / unresolved | Pair scan P95 (µs) | Owner postings P95 (µs) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 23 | 1 | 13 | 22 / 1 | 10.12–10.25 | 15.53–15.80 |
| 16 | 40 | 1 | 18 | 37 / 9 | 28.20–28.39 | 30.64–30.75 |
| 32 | 61 | 1 | 20 | 46 / 20 | 89.77–91.11 | 60.51–60.69 |
| 64 | 104 | 1 | 28 | 49 / 29 | 323.29–326.09 | 109.55–110.54 |

At depth 64 the postings implementation selects the same three
sources at about one-third of the reference component P95.
At eight or sixteen, use the smaller pair scan for this workload.
Initial exposure remains a limitation: keeping the common first
two papers and allowing one more reaches only 29 requests at depth
64. Joint nomination reaches 49, while another 55 requests
contain a reference proof in the pool that the three reads and
output rule fail to recover. Publisher topic tags remain incomplete.
The additional unresolved results at deeper cuts require independent
review before trading off width and risk. A benchmark on an independent
catalog and full request measurements are still needed.

[Aggregates and timings](results.json)

## Reproduce

Use the same publisher XML files and verified Qwen3 checkpoint as
the open topic-and-role benchmark. Run within a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.joint_byline_depth \
  benchmarks/plos-credit-cross-source-v1 QWEN_EMBEDDING NEW_RESULTS XML_DIRECTORY
```

Repeat with `--order-offset 1`.
