# Replan evidence reads from observed contributor roles

Use the archived 452-article mixed catalog and the same 160 PLOS and
51 eLife topic-role requests as the [matched-role control](../open-topic-role-matched-v1).
Freeze the top 128 hybrid-ranked articles per clause. Keep the original
declaration verifier and read exactly three distinct sources per route.
Run at 100%, 50%, and 25% deterministic exposure of role metadata.
Sources hidden from the role index remain available for full reading.

| Route | Reads and decision |
| --- | --- |
| Sequential | Independent anchors followed by the archived owner-gap action |
| Joint | Plan all three from indexed roles, then open together |
| Split joint | Same plan, opened in a two-source batch followed by a single read |
| Repair third | Start with the joint anchors and consider replacing the last read after observing them |
| Repair second and third | Start with the first joint anchor and consider replacing each subsequent read |
| Prefer indexed repair | Also replace an opaque scheduled source if a confirmed anchor can pair with a source listing the missing role |

Preserve the scheduled source once the read prefix establishes a proof.
Otherwise preserve a candidate source compatible with a confirmed
anchor, subject to the opaque-source replacement rule in the last arm.
If a repair is possible, prefer indexed owner-role matches to opaque
ones, then use the minimax and summed ranks of the two sources. Preserve
the scheduled choice when no replacement satisfies the rule. Read only
the selected sources. Publisher subject tags are used only after the
response is produced.

Compare naturally issued outputs and the matched panel-wide issuance
quota. Record the number of labeled proofs reachable within the reads,
the chosen proof, and warm local P95. Require the complete-index
sequential and joint controls to reproduce their archived signatures.
Require split joint to return the same plans and proofs as joint on
every request. Repeat the sweep in reverse request and arm order.

## Results

Slurm job 47542 completed both arms orders on an RTX 4090 on ws6.
All six panels reproduced their decision signatures. The three
adaptive routes produced the same returned expert and citations
as the static joint route for every request. Splitting joint reads
does not change any plan. The last-read repair never changes a read.
For each masked PLOS panel, both two-stage variants replace the
second source once and preserve the final result. Full-index plans
and all eLife plans stay identical.

| Panel | Role index | Known / issued, joint and all repair arms | Known proof reachable within three reads |
| --- | ---: | ---: | ---: |
| PLOS | 100% | 85 / 140 | 89 |
| PLOS | 50% | 56 / 114 | 59 |
| PLOS | 25% | 32 / 68 | 33 |
| eLife | 100% | 46 / 49 | 46 |
| eLife | 50% | 35 / 45 | 35 |
| eLife | 25% | 34 / 42 | 34 |

Full-index warm local P95 ranges from 21.7 to 23.9 ms across routes,
panels and both execution orders. Some masked runs exceed 30 ms
across routes. The timing does not represent remote read waits.
The evidence shows little opportunity for this action rule to change
plans, even with a sparse role index. It supports keeping static
joint planning for this benchmark and testing new planning signals
before adding serial reads. The same requests are reused across masks.

[Aggregate results and replication hashes](results.json)

## Reproduce

Restore the frozen XML and the open checkpoint specified by the
[matched-role benchmark](../open-topic-role-matched-v1#reproduce).
Repeat with `--reverse` in a CUDA Slurm allocation.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.open_topic_observed_repair \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json QWEN_EMBEDDING \
  benchmarks/mixed-publisher-joint-v1/results.json \
  benchmarks/live-joint-source-request-v1/results.json \
  benchmarks/open-topic-role-matched-v1/results.json NEW_RESULTS.json
```
