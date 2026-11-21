# Certify a fixed proof plan before source reads

Use the two disjoint eLife projections, the query constructor, the
open 0.6B embedding, and the packed global role index from the
[fixed exact-index comparison](../linked-role-lane-index-v1).
Construct the same three-source joint plan at each candidate depth.
Run once with all parsed role data visible and once after withholding
role rows by a stable source hash (nominal 50%). Compute a separate
three-source reference within each visibility condition. Withholding
roles can change the plan. Every source remains readable.

| Route | Source reads |
| --- | --- |
| Three-read reference | Open the entire fixed plan together. |
| Fixed two | Open the first two sources and verify. Included as a strong uncertified control. |
| Post-read certificate | Open the first two. Open the third only if its optimistic completion can change the verified response. |
| Pre-read certificate | Form the optimistic proof set using known and possible roles within the plan. Abstain without opening a file if this set is empty. If the most preferred known and optimistic identity–citation tuples coincide, read just its two citations in one wave. Otherwise open the entire plan in one wave. |

The best proof uses worst lane rank, summed ranks, contributor, and
the two source IDs, in that order. The exactness argument assumes a
fixed byline, a fixed source inventory, and indexed known roles that
agree with the original declaration. Omitted role information remains
possible. A positive pre-read certificate rereads the selected
declarations and completes the third read if verification disagrees.
An untrusted or changed index requires the reference route. There is no
claim about proof search outside the three planned documents.

## Results

Slurm job 47566 completed on ws6 with an RTX 4090. Erasing timing
fields gives identical results and per-request decision signatures in
both orders. At full visibility the reference reproduces the issued
and publisher-tag-supported counts of the archived exact-index study.
The two certificate arms reproduce every reference person, ordered
citation and abstention. The fixed-two control also happens to match
in every condition. There are 398 requests per table row, repeated
across depth and visibility.

| Depth | Role visibility | Three reads | Fixed two / post-read | Pre-read | Requests with no reads under pre-read | Pre-read waves |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 100% | 1,194 | 796 | 88 | 354 | 44 |
| 32 | 50% | 1,194 | 796 | 128 | 348 | 50 |
| 128 | 100% | 1,194 | 796 | 526 | 136 | 262 |
| 128 | 50% | 1,194 | 796 | 791 | 100 | 298 |

The pre-read route does not need an extra wave in either order.
The strongest efficient control here is fixed two. At depth 128
with withheld roles, the pre-read route saves just five more reads
overall and opens 50 more declarations on the confirmation snapshot.
Its value at that operating point is fewer requests reaching the
source service, a load hypothesis. At full visibility the reduced work
comes mainly from graph-certified abstentions. At depth 32 the reference
has 44 issuances; at depth 128 it has 260 with complete role data.
Changing depth changes the operating point. Timing comparisons are
made within each depth and visibility. Across the original conditions
the pre-read minus full warm local P95 difference ranges from about
−0.11 to +0.02 ms. The timer includes the shared encoder and ranker
but excludes network, queueing and corpus startup. This replay
establishes exact read savings on the structured benchmark.

[Both replay receipts](results.json)

## Reproduce

Restore the [frozen eLife projections and open checkpoint](../linked-role-lane-index-v1#reproduce).
Use a Slurm CUDA allocation. Repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.pre_read_proof_bound \
  EXTENDED_PROJECTION benchmarks/elife-extended-role-holdout-v1/manifest.json \
  CONFIRMATION_PROJECTION benchmarks/elife-source-guard-confirmation-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/linked-role-lane-index-v1/results.json NEW_RESULTS
```
