# Learn a no-extra-read role decision from a proof-gated reader

Reuse the [guarded single-clause recheck](../guarded-role-recheck-v1).
Keep the anchored source retrieval, ridge and prototype role bands,
proof-aware slot selector, and the prototype agreement guard. The
4B reference chooses a role from the unresolved ridge band plus an
abstention option. The replacement sees the same options but reuses
the role embedding already computed for the request. No candidate,
author identity, or source text enters either role selector.

For each of the 84 prior training phrases, create three option sets:
the full glossary, the ridge band augmented with the target and
second-ranked role, and the three highest-ranked roles with the
target omitted. In the omitted condition the label is abstention.
Run the same Qwen3-4B-Instruct-2507 decision model on all three sets.
Fit two 15-output linear classifiers on frozen Qwen3 embeddings
using masked cross-entropy. The label-only route uses the known
synthetic role. The distilled route equally weights the synthetic
label and the teacher's hard decision. Both use the same
initialization, optimizer, and 180 steps. The projection backbone
is frozen. Compare also an untrained average of standardized
ridge and prototype scores over eligible roles.

Calibrate each replacement's confidence gap on the previous 48
PLOS role cases, maximizing strict proofs under 5% observed adverse
cases. No eLife or engineering requests or labels are used to fit
weights or thresholds. Their independently authored NISO phrases
are text-disjoint from the 84 training phrases. The two panels are
reused here to compare the reader replacement with its previous
reference. Controls hide each target source before retrieval.

## Results

The teacher agrees with the synthetic target in 184 of 252 option
sets. Each fitted head has 15,375 parameters. All three replacement
routes select a zero gap cutoff on PLOS, each with 24 strict proofs
and zero observed adverse cases on those calibration groups.

The main table uses the excerpt queries. An adverse output includes
an incorrect contributor or requested role. Missing-source views
are a separate control. Latency is the warm local query path.

| Route | eLife proofs / 51 | eLife adverse / 102 | eLife substitutions / 204 missing views | eLife P95 ms | Engineering proofs / 15 | Engineering adverse / 30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static prototype band | 23 | 0 | 0 | 12.8–12.9 | 4 | 0 |
| Guarded 4B teacher | 30 | 0 | 0 | 34.5–34.9 | 5 | 0 |
| Dense agreement | 26 | 5 | 1 | 12.9 | 3 | 5 |
| Label-only head | 24 | 1 | 1 | 12.9 | 2 | 0 |
| Distilled head | 26 | 3 | 1 | 12.8–12.9 | 2 | 4 |

On eLife keywords the static, teacher, dense, label-only and distilled
routes find 16, 23, 20, 18 and 20 strict proofs respectively. The three
replacements have 5, 1 and 3 adverse outcomes. None of the five routes
uses a substitute on keyword missing-source views. Engineering keyword
proof and adverse counts equal the engineering excerpt counts.

The student restores static-route latency and has no 4B calls.
Relative to label-only learning, distillation gains two strict eLife
proofs and adds two adverse outcomes in each clue condition. On the
engineering panel it adds four adverse outcomes without improving the
label-only proof count. The favorable PLOS calibration did not expose
these errors. A contender needs more diverse negative controls and
uncertainty calibration before replacing the teacher. In particular,
source substitution and wrong-role issuance must remain separate
endpoints.

Both Slurm tasks in array 45942 ran on ws3 RTX 5090. All non-timing
fields match with reverse query order. The teacher and static results
also match the preceding guarded study. Local factored latency
includes the live embedding batch, source retrieval and role decision
with its guard. It excludes initial index construction, network
transport and unconstrained request parsing. The two target panels
reuse paper pairs and role phrases. [Full results](results.json)

## Reproduce

Use the pinned eLife and engineering projections and the earlier
PLOS calibration projection. The metric file records both model
revisions. Run inside a Slurm CUDA allocation with a fresh output
path and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.distilled_role_recheck \
  ELIFE_PROJECTION ENGINEERING_PANEL PLOS_PROJECTION NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```
