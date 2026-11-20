# Diagnose candidate versus evidence headroom on enterprise requests

Reuse the 25 development requests with parsed qualifications in the
[enterprise joint first-round comparison](../enterprise-joint-first-round-v1).
All 25 initially abstained in the hybrid-prefix control. Use the pinned
qualifications and repeat live query embedding, keyword retrieval and
expert fusion. The first five ranked people and their original two
excerpts must match the frozen trace. Reproduce the original verifier
batch order on those excerpts and verify the same 25 abstentions.

Cross two factors. The people pool is either the first 5 or first 20
from the same ranking. The evidence pool is either the original
first two excerpts or up to eight excerpts per person. Expanded
evidence keeps both original excerpts and draws additional candidates
from deeper source prefixes and the indexed personal-role excerpts.
Round-robin lexical ranking per condition gives each condition a
chance to nominate an excerpt. All additional condition–excerpt
pairs are scored once and reused across the diagnostic arms.
Report the ranks 6–20 separately.

A person is provisionally complete if every parsed condition has
some positive model margin among that person's inspected excerpts.
This is a diagnostic of verifier-positive opportunity, rather than
human relevance, completeness, or validated attribution. Report
topics and qualifications separately. Budgets differ deliberately.
An expanded method can miss suitable experts or evidence outside its
pool, and a positive margin can be wrong. This experiment fixes
the parser to isolate retrieval and verifier behavior; it does not
measure end-to-end latency.

## Development results

Both tasks of Slurm array `44820` on `ws5` reproduced all outcome
counts and every provisional person and exact citation.

| People and evidence pool | Expert–request opportunities | Inspected excerpts | Topic-positive people | All qualifications positive | All conditions positive | Requests with a provisional expert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First 5, original two | 125 | 250 | 26 | 2 | 0 | 0 |
| First 20, original two | 500 | 997 | 63 | 7 | 4 | 2 |
| First 5, up to eight | 125 | 869 | 34 | 4 | 3 | 1 |
| First 20, up to eight | 500 | 3,344 | 83 | 11 | 7 | 2 |
| Ranks 6–20, up to eight | 375 | 2,475 | 49 | 7 | 4 | 2 |

Expanding people alone finds four provisional people on two
requests. Expanding excerpts alone finds three people on one of
those requests. Combining the two changes finds seven across the
same two requests. The other 23 remain unanswered by this verifier
and pool. At the broadest setting, topic evidence passes
for at least one person on 20 requests, but all qualifications
pass for at least one person on only four. Consequently the
qualification check deserves separate study from topical retrieval.
This evidence cannot distinguish genuinely absent qualifications
from unindexed source spans or false rejections.

The widest path inspects about 13.4 times as many excerpts as the
original five-person policy. It is diagnostic and cannot supply
a same-budget algorithm claim. Its seven provisional people form
seven exact-span cards in two local reviewer copies. Both copies
validate against the original sources and are unjudged.

[Aggregate screen results](results.json).

## Reproduce

Use the private trace and raw aggregate from the original first-round
run, with the pinned development corpus, role window and vector
artifacts. Run on a Slurm GPU. Use `--order-offset 1` to reverse
query order for the replicate. Keep trace and output containing
identities and citation spans outside Git.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_joint_headroom \
  DEV_QUERIES DEV_CORPUS ROLE_WINDOWS VECTORS.npy VECTOR_METADATA CHECKPOINTS \
  SCRATCH PRIVATE_FIRST_ROUND_TRACE FIRST_ROUND_RUN_AGGREGATE NEW_AGGREGATE PRIVATE_OUTPUT
```
