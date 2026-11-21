# Locate first-query failures under a shared verifier budget

Reuse the 108 structured requests and public declarations from the
[eLife loopback study](../paired-role-prospective-service-v2). Freeze
candidate generation, source confirmation, role shortlist, and the
PLOS-calibrated batched reader threshold. Check that the strict
agreement control reproduces every evidence response from the
[archived lightweight replay](../lightweight-paired-role-v1).

All policies may spend at most one source or role model forward.
Preserve the frozen single-clause repair and all source-stage
responses. For an unresolved source pair with the forward still
available, use the author-role index to check whether a feasible
role pair exists before a batched role read. Test three acceptance
policies over the same three alternatives per activity.

- **Exact agreement.** Both reader labels must equal their dense leaders.
- **One disagreement.** At most one reader label may differ from its dense
  leader. The other retains agreement.
- **Any feasible pair.** Both reader labels may differ.

All policies require the selected roles to be declared for the same
contributor on the respective papers. The role reader sees only the
activity texts and candidate role definitions. The replay can share
a batched observation across policies. Account for each policy's
own forward and token use independently.

For missed positives in the agreement control, identify whether the
correct source pair was unresolved or replaced, the source stage had
already used the forward, or the role stage could not complete the
proof. These diagnostics use the published labels only after the
policy selects its output. Replay in both request orders. This
analysis covers structured requests with the target papers present.
It supplies no service timing or evidence for raw enterprise inputs.

## Results

Both tasks in Slurm array 46226 completed on ws3 RTX 5090.
The two orders reproduce identical results and evidence signatures.
The exact agreement control matches the archived service proof for
all 108 requests.

| Acceptance policy | Keyword strict / adverse | Excerpt strict / adverse | Role forwards across 108 requests | Role prompt tokens |
| --- | ---: | ---: | ---: | ---: |
| Exact agreement | 12 / 0 | 18 / 0 | 22 | 2,667 |
| One disagreement | 13 / 1 | 19 / 2 | 47 | 6,447 |
| Any feasible pair | 15 / 1 | 21 / 2 | 59 | 8,251 |

Each view has 27 positives and 27 role-swap controls. Source forwards
stay at four, all on the keyword view. On keyword requests, eight of
the 15 missed positives never reached a selected source pair.
On excerpt requests, every missed positive reached the correct
source pair. Five of the nine excerpt misses disagreed with at least
one dense leader in a batched read, three lacked a feasible pair in
the three-option pools, and one failed after the single-clause read.
A disagreement can still produce the wrong role or contributor.

Permitting one disagreement adds two strict proofs and three adverse
recommendations at more than twice the role-forward count. The
broader gate adds six strict proofs and three adverse recommendations.
Keep exact agreement as the measured low-risk reference. The failure
split motivates source recovery for keyword clues and better role
candidate representations for excerpts. The alternatives here reuse
an explored public panel. Their quality-cost results require a
prospective replication. See [aggregate results](results.json).

## Reproduce

Restore the [prospective replay inputs](../paired-role-prospective-service-v2#reproduce)
and the corresponding single-run cascade result. Use a fresh output
inside a Slurm GPU allocation. Repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.credit.paired_role_prospective_service \
  ELIFE_PROJECTION ELIFE_MANIFEST benchmarks/paired-role-verification-v1/results.json \
  NISO_EXAMPLES ROLE_TRAINING SCORE_BANDS SOURCE_FROZEN ROLE_FROZEN \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_Both tasks in Slurm array 46226 completed on ws3 RTX 5090.
The two orders reproduce identical results and evidence signatures.
The exact agreement control matches the archived service proof for
all 108 requests.

| Acceptance policy | Keyword strict / adverse | Excerpt strict / adverse | Role forwards across 108 requests | Role prompt tokens |
| --- | ---: | ---: | ---: | ---: |
| Exact agreement | 12 / 0 | 18 / 0 | 22 | 2,667 |
| One disagreement | 13 / 1 | 19 / 2 | 47 | 6,447 |
| Any feasible pair | 15 / 1 | 21 / 2 | 59 | 8,251 |

Each view has 27 positives and 27 role-swap controls. Source forwards
stay at four, all on the keyword view. On keyword requests, eight of
the 15 missed positives never reached a selected source pair.
On excerpt requests, every missed positive reached the correct
source pair. Five of the nine excerpt misses disagreed with at least
one dense leader in a batched read, three lacked a feasible pair in
the three-option pools, and one failed after the single-clause read.
A disagreement can still produce the wrong role or contributor.

Permitting one disagreement adds two strict proofs and three adverse
recommendations at more than twice the role-forward count. The
broader gate adds six strict proofs and three adverse recommendations.
Keep exact agreement as the measured low-risk reference. The failure
split motivates source recovery for keyword clues and better role
candidate representations for excerpts. The alternatives here reuse
an explored public panel. Their quality-cost results require a
prospective replication. See [aggregate results](results.json). \
  --stage-audit-reference CASCADE_RUN
```
