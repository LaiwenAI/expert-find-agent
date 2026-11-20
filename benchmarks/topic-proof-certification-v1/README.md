# Verify ranked three-condition proofs without scoring every topic edge

Retain the [three-qualification workload](../multi-qualification-proofs-v1)
and the fixed BM25/dense ranks at depth 128. Each candidate proof
binds one declared person to three ordered topic–role conditions,
with the active source-separation rule. The two forms require
three distinct sources or require only the first two sources
to differ. Enumerate all structurally feasible proofs in the
original rank order. A topic edge connects a condition to a
source. A source reused in two conditions requires two separate
positive topic judgments. Complete role declarations are assumed.

Use the same open-weight Qwen3-Reranker-0.6B on the condition's topic
and the article title and abstract for every method. Each threshold
decision is cached within the request and shared across all plans
that use its edge. Reread the returned source declarations to
verify the same contributor and role for each exact citation.
Publisher topic annotations are used only for scoring after output.
Unannotated results remain unresolved.

## Schedules

| Schedule | Topic verification |
| --- | --- |
| All edges | Verify every edge occurring in a feasible proof, with at most 12 pairs per model batch. |
| Static 9 | Use at most nine different edges in ranked-proof order, in a single batch. Return the best verified proof among those edges. |
| Adaptive 9 | Use the same overall cap. Test the missing edges of the top surviving proof, discard plans that include rejected edges, and repeat if the remaining budget can complete the next proof. |
| Certificate | Apply the adaptive schedule without a pair cap. Stop at the first surviving plan with every edge verified, or when all plans are rejected. |
| Prefetch 9 | Apply exact stopping while fetching up to nine unseen non-rejected edges in ranked-proof order per wave. This speculates on subsequent plans to limit model calls. |

Exact stopping follows from the fixed proof order. Every higher plan
contains a rejected edge once the scheduler stops at a verified plan.
We test the schedule with common all-edge signs and separately with
live inference for each arm. Batch shape can change a model's numeric
sign near the threshold, so report observed disagreements rather than
assuming identical neural decisions. Warm local P95 includes query
encoding, retrieval, proof enumeration, topic verification, and
original declaration checks. It excludes serving queues and network
transport. These panels are constructed and share sources.

## Results

Slurm array `45719` completed four times on the ws3 RTX 5090,
with two repeats per cutoff and rotated policy order. Each cutoff
reproduced all non-timing outcomes. The previous role-only route
issues 38 results across 48 constructed requests, of which 27 have
all three topic annotations. At the default margin cutoff of zero,
only 18 of 138 annotated topic edges pass. No method returns a
complete proof at that cutoff. This does not provide a useful
quality comparison of the schedules.

The sensitivity run uses a preset margin cutoff of −4. There is no
learned or test-optimized calibration. At this cutoff 83 of 138
annotated edges and 16 of 75 unjudged edges pass under the full
reader. Each schedule returns the same six ordered proofs, all from
PLOS. Four have all three topic annotations. The remaining two need
independent assessment. No observed topic signs change across GPU
batch shapes at either tested cutoff.

| Schedule at −4 | Topic pairs scored | GPU forwards | Complete proofs | Known proofs |
| --- | ---: | ---: | ---: | ---: |
| All edges | 213 | 40 | 6 | 4 |
| Static 9 | 194 | 38 | 6 | 4 |
| Adaptive 9 | 156 | 55 | 6 | 4 |
| Certificate | 162 | 57 | 6 | 4 |
| Prefetch 9 | 197 | 39 | 6 | 4 |

The exact sequential certificate reduces pairs by 23.9% relative to
all-edge scoring, while adding 17 model forwards. On the PLOS form
that permits reuse, the same-cutoff warm request P95 ranges are
71.16–71.79 ms (all edges), 52.37–52.86 ms (static nine), 49.66–50.36 ms
(adaptive nine), and 63.70–64.41 ms (exact sequential certification).
On the corresponding eLife form, static nine is faster than adaptive
nine despite scoring four more pairs. The exact schedule certifies
a fixed model's output, which can still miss genuine matches.
A single-batch static baseline remains important. The next quality
test needs independent topic judgments for actual requested
conditions and calibration on a separate corpus.

[Aggregate results](results.json).

## Reproduce

Restore both frozen XML inputs and the two prior results. Run on a
Slurm-allocated CUDA GPU and rotate the arm order for replication.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.topic_proof_certification \
  PLOS_XML PLOS_FIXTURE ELIFE_XML ELIFE_MANIFEST QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/live-joint-source-request-v1/results.json \
  benchmarks/multi-qualification-proofs-v1/results.json NEW_RESULTS --cutoff=-4
```
