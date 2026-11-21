# Bound the sources that can change a fixed-plan proof

The [endpoint certificate](../pre-read-proof-bound-v1) reads all three
planned declarations when unknown roles let a different proof outrank
the best fully indexed proof. This comparison asks which files can
affect the result under those unknown roles. The best proof from known
roles is a valid incumbent under a version-matched index. For every
oriented pair of distinct sources, allow all unknown contributor roles
to match and compute that pair's best possible rank key. Retain both
files if the pair can tie or improve on the incumbent. If there is no
incumbent, retain both files in each feasible pair. Open the retained
files in one wave and run the unchanged original declaration verifier.

Any winning proof belongs to an optimistic feasible pair. Its rank
cannot be worse than the known incumbent when one exists. Its two
files are therefore retained. This also preserves abstention and exact
citation ties under the original total rank order. It relies on the
same fixed plan, author identity and known-role integrity assumptions
as the endpoint certificate. For every compared request, require that
the frontier opens no more files than the prior endpoint policy and
matches the original unpruned output after reading.

Use both frozen eLife snapshots at depth 128. The four panels reuse 398
requests over two index visibility conditions. The 50% condition hides
role rows by a stable source hash while keeping byline identities
and every declaration available. A full and a two-read reference are
defined inside each condition.

| Nominal role visibility | Fixed three | Fixed two | Endpoint bound | Source frontier | Further reads skipped |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 1,194 | 796 | 526 | 525 | 1 |
| 50% | 1,194 | 796 | 791 | 604 | 187 |

Slurm job 47569 on ws6 uses the pinned open embedding. Both traversal
orders reproduce every aggregate and per-request decision hash. Both
certificates reproduce every person, citation and abstention. At 50%
visibility the frontier avoids 187 of 791 endpoint reads (23.6%).
This measures local declaration access. Dispatch CPU time is included
separately in the receipt; no service latency is inferred.

[Local replay receipts](results.json)

## Source-service comparison at 50% visibility

Slurm job 47570 repeats the exact comparison over the six-slot
[loopback source service](../pre-read-proof-http-v1), with its fixed
16 ms source delay. Time starts at scheduled arrival with ranks and
plans frozen. Both orders reproduce every response byte, declaration
count, and transfer size. Each cell is P95 in milliseconds.

| Scheduled arrivals/s | Fixed two | Endpoint bound | Source frontier |
| ---: | ---: | ---: | ---: |
| 165 | 19.50–20.70 | 49.39–64.46 | 21.81–22.83 |
| 205 | 302.99–324.45 | 300.68–317.35 | 35.48–36.63 |

At 165/s the frontier trims the endpoint policy's tail, with fixed
two still a little faster. At 205/s both controls repeatedly queue
at the source bottleneck. Reducing the read count from 796 or 791
to 604 lowers offered source traffic enough for the frontier to
largely keep up. These are finite source-service traces. They
exclude parsing, live retrieval and external storage. Repeat the
reproduction command with `--http` to include both load settings.

[Source-service receipts](http-results.json)

## Reproduce

Restore the pinned inputs from the
[pre-read benchmark](../pre-read-proof-bound-v1#reproduce). Run on a
Slurm CUDA allocation and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.proof_source_frontier \
  EXTENDED_PROJECTION benchmarks/elife-extended-role-holdout-v1/manifest.json \
  CONFIRMATION_PROJECTION benchmarks/elife-source-guard-confirmation-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/pre-read-proof-bound-v1/results.json NEW_RESULTS
```
