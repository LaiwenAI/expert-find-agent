# Condition-aware expert nomination under a fixed verification budget

Replay the 80 pinned development requests and 52,667 source records
from the [first-round study](../enterprise-joint-first-round-v1).
Use the exact same qualification spans from the frozen development
trace. At most five people and two excerpts per person reach the
verifier. Preserve the original excerpt quota for every slot.
The original hybrid control must reproduce its shortlist, exact
provisional citations and verification pair count on every request.

For a multi-condition request, keep the first two hybrid nominees.
Within the first 50 hybrid nominees, batch-encode the topic and
qualification clauses. Within each condition, score a person's
strongest owned source, then rank people on that condition. Fill
the remaining slots by minimizing the worst condition rank, then
the sum of ranks and original position. For every nominee, compare
the original hybrid source prefixes against joint selection from
owned windows. The shortlister falls back to the original pool
when there are no admissible qualification clauses. It does not
discard criteria from verification.

The verifier, original source offsets, checkpoint revisions,
condition set, and verification pair count stay fixed. Timings
start after qualification parsing and cover live full-query
embedding, retrieval, any condition-specific reranking and
verification. They do not include the original qualification
parser, model loading, or serving. Alternating order controls
potential warm-run effects.

## Results

Both Slurm tasks in array `45196` reproduce every shortlist,
selected excerpt and provisional citation. All arms make 1,279
verification checks with no clipped inputs.

| Nomination and source selection | Issued requests / 80 | Issued people | Multi-condition requests issued / 25 | Changed multi-condition shortlists | Post-parse P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid, prefixes | 25 | 51 | 0 | 0 | 263.6–265.4 ms |
| Hybrid, joint windows | 25 | 51 | 0 | 0 | 265.3–265.9 ms |
| Per-condition, prefixes | 26 | 52 | 1 | 25 | 257.8–259.7 ms |
| Per-condition, joint windows | 26 | 52 | 1 | 25 | 258.5 ms |

The additional provisional recommendation is the same expert
and exact evidence view in both condition-aware arms. It is
the sole changed view, and has been retained in a private
blind-review packet with no completed judgments. Even if support
is confirmed, the observable gain in supported P@5 on these
80 requests is at most 0.25 percentage points. The timing ranges
are descriptive. The results concern frozen development clauses
and a model verifier. They do not estimate human accuracy or
the effect of searching beyond the top 50 hybrid candidates.
[Aggregate results](results.json).

## Reproduce

Run on a Slurm CUDA allocation with the pinned development
inputs and exact trace from the original first-round comparison.
The two output paths must be new. Keep the per-request trace private.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_condition_nomination \
  QUERIES CORPUS ROLE_WINDOWS VECTORS VECTOR_META MODELS SCRATCH \
  FIRST_ROUND_AGGREGATE FIRST_ROUND_PRIVATE_TRACE NEW_AGGREGATE NEW_PRIVATE_TRACE
```

Repeat with `--order-offset 1`.
