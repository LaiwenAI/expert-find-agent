# Calibrate role options without using source declarations

The [fixed-pair menu comparison](../elife-fixed-pair-role-menu-v1)
found that the 4B reader missed many recoverable proofs even with all
14 CRediT roles on the menu. Test whether a role-name or option-position
prior explains the misses. Keep both selected papers, the byline
links, all role names, the frozen margin threshold, source masking,
and model revisions unchanged. The role reader sees the two clauses
independently and never sees the documents when classifying them.

The offline calibration adapts the content-free probe idea in
[Zhao et al. (ICML 2021)](https://proceedings.mlr.press/v139/zhao21c.html).
Use three neutral strings and the 14 cyclic rotations of the role
menu. Center each probe's 14 role logits. Every role visits every
position once per string. Average by position and by role to estimate
two additive offsets. The no-match option stays last and its score is
unchanged by either correction. The calibration uses 42 rows and
5,040 unpadded input tokens offline, without held-out example text,
gold roles, or outcome feedback.

| Route | Change |
| --- | --- |
| Full-menu reference | Use the original deterministic, phrase-keyed option order |
| Position correction | Subtract the offline position offset |
| Role correction | Subtract the offline role-name offset |
| Joint correction | Subtract both offsets |
| Reversed order | Reverse the role positions and decode in one forward |
| Two-order average | Align raw logits by role and average the original and reverse-order outputs |

The three corrections reuse the reference logits and add no reader
forwards or tokens per request. Reversed order uses one forward and
the average uses two. Token counts describe logical prompts and the
experiment does not measure latency.

## Results

With both papers available, the three one-forward corrections reproduce
the original route's strict, adverse and missing-source counts. Every
grouped paired interval for strict proofs is [0, 0]. The estimated
position offsets span about 5.48 logits, with no measured gain in
strict proofs in either view.

| Route | Confirmation keywords, strict / adverse | Extended keywords, strict / adverse | Confirmation excerpts, strict / adverse | Extended excerpts, strict / adverse |
| --- | ---: | ---: | ---: | ---: |
| Full-menu reference | 29 / 4 | 39 / 8 | 38 / 8 | 53 / 12 |
| Each offline correction | 29 / 4 | 39 / 8 | 38 / 8 | 53 / 12 |
| Reversed order | 29 / 3 | 38 / 7 | 37 / 7 | 54 / 9 |
| Two-order average | 28 / 4 | 38 / 8 | 36 / 8 | 54 / 11 |

The panels have 51 and 72 positive requests and equally many controls.
A strict proof requires the exact source pair, a gold shared author,
and both intended roles. Any issued proof that fails this definition is
adverse. Reversing the positions reduces adverse counts by one to three
across the four comparisons and preserves or changes the strict count
by one. All four paired strict-outcome intervals include zero. The
two-order average doubles model forwards and recovers no net strict
proofs on keywords. When one intended paper is withheld, every route
has zero keyword substitutions and four to five excerpt substitutions.

The measured recognition gap survives position and role-name correction
under this prompt and checkpoint. Reversal is a sensitivity diagnostic.
A quality claim would require improvement that persists on independently
worded requests. The two panels reuse phrases from the same pinned NISO
glossary, and their intervals resample paper/author groups.

Slurm array 46623 ran all four tasks on ws3 RTX 5090. Forward and reverse
case traversal reproduce the result signatures and calibration offsets.
The original route outcomes match the archived run.

## Reproduce

Use the pinned confirmation and extended panels, NISO example snapshot
and open Qwen checkpoints from the
[fixed-pair menu comparison](../elife-fixed-pair-role-menu-v1#reproduce).
Run its inference command with `--calibrate-options` in place of
`--menu-ablation`. Run each panel forward and reverse in a Slurm CUDA
allocation. Reuse the existing aggregation command to compare
against the archived reference and to verify order replication.

[Full metrics, intervals and calibration offsets](results.json)
