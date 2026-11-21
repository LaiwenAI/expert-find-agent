# Shorten role prompts while keeping the proof checks fixed

Reuse the [guarded two-source recheck](../guarded-role-recheck-v1)
on the same eLife and engineering panels. Fix source retrieval,
candidate role bands, choice of clause to recheck, independent
prototype support check, stance on missing sources, and the
previous Qwen3-4B-Instruct-2507 margin cutoff. Change only the
rendered CRediT options in the reader prompt.

- **Full:** role name and the original NISO definition.
- **Compact:** role name and a manually written single-phrase
  description for each of the 14 roles, defined in the runner.
- **Name only:** role name without a definition.

Every version includes the same none-of-these option. Option order
is identical for a given phrase and candidate set. Each live reader
runs under its own timed interval. Execution order rotates by
request, and the second task reverses query order. No thresholds
are fitted on these panels.

## Results

The table uses excerpt queries. The prompt counts sum the actual
input tokens of all calls, including the task phrase and chat
template. An adverse result with both papers available is a
wrong contributor or a wrong cited role. Substitutions are
measured separately with either named paper hidden.

| Panel | Prompt | Strict proofs | Adverse | Missing-source substitutions | Input tokens | Local P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| eLife | Static prototype | 23/51 | 0/102 | 0/204 | 0 | 12.9 ms |
| eLife | Full | 30/51 | 0/102 | 0/204 | 4,466 | 34.3–35.3 ms |
| eLife | Compact | 28/51 | 1/102 | 1/204 | 2,969 | 27.2–27.4 ms |
| eLife | Name only | 29/51 | 1/102 | 1/204 | 1,999 | 24.0–24.1 ms |
| Engineering | Static prototype | 4/15 | 0/30 | 0/60 | 0 | 12.7–12.8 ms |
| Engineering | Full | 5/15 | 0/30 | 0/60 | 2,682 | 38.1–38.2 ms |
| Engineering | Compact | 3/15 | 0/30 | 0/60 | 1,558 | 27.6–27.9 ms |
| Engineering | Name only | 6/15 | 0/30 | 0/60 | 883 | 25.1–25.2 ms |

On keyword queries, the full, compact, and name-only routes retain
23, 22 and 23 of 51 eLife positives. In engineering they retain 5,
3 and 6 of 15. The two shortened eLife prompts each issue one swap
output with both papers available. No route emits a substitute in
the keyword missing-source views. All three reader versions make
22 excerpt and 17 keyword calls on eLife, and seven calls per view
on engineering.

The name-only prompt cuts 55% of the eLife excerpt input tokens
and brings tail latency to about 24 ms. It loses one eLife proof
and introduces both observed error types compared with the full
prompt. Engineering gains one proof with the same number of
observed errors. Compact definitions lose more positives than
the names in both panels. The full prompt remains the
quality-oriented comparison. Shorter prompts warrant independent
validation with ambiguous role wording before an operating point
is selected.

Both tasks of Slurm array 45947 ran on ws3 RTX 5090 and reproduce
all non-timing fields. Full-prompt and static counts match the
previous guarded study. The factored latency charges each route
for the initial live embedding, anchored search, guards and its
reader call. It excludes build, transport and unconstrained
request parsing. The panels reuse cases within four engineering
and ten eLife paper pairs. [Full results](results.json)

## Reproduce

Restore the pinned eLife and engineering projections. Use the
model revisions from the metric file. Run inside Slurm CUDA
with a fresh output path and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.compact_role_prompts \
  ELIFE_PROJECTION ENGINEERING_PANEL NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```
