# Change the role menu while holding both sources fixed

Extend the [fixed-source role recognition comparison](../elife-role-feedback-reads-v1).
Reuse its confirmation and extended eLife panels, two-read guarded
same-author pairing, open Qwen 0.6B embeddings, Qwen3-4B reader, and
frozen margin cutoff. Each clause has its own role recognition row
inside the same 4B forward. The reader never sees the selected
sources or their role declarations.

| Menu | Options presented to the reader |
| --- | --- |
| Three per slot | Top three CRediT role names per clause from the sum of the standardized ridge and prototype scores |
| All per slot | All 14 role names for both clauses |
| Expand one slot | All 14 role names for the clause with the smaller top-two score margin. Top three for the other clause. Ties expand the first clause. |
| Explain all | All 14 role names followed by the [existing brief definitions](../../experiments/credit/compact_role_prompts.py) for both clauses |

All menus keep the no-match option and deterministic ordering. Menu
choices use neither the retained source declarations nor held-out
labels. A strict proof requires the exact intended pair, a gold
shared author and both intended roles. An adverse output is an issued
proof that fails this definition. The menu ceiling counts positive
queries where the fixed source pair is exact and both intended roles
are allowed. Record whether raw role decoding and thresholded
role decoding each recover the intended roles.

## Results

Keyword view with both papers available. Input tokens count logical,
unpadded reader tokens over eligible positive and control requests.

| Panel | Menu | Ceiling | Strict proofs | Adverse outputs | Input tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Confirmation | Three per slot | 37 | 27/51 | 5/102 | 11,994 |
| Confirmation | All per slot | 39 | 29/51 | 4/102 | 20,644 |
| Confirmation | Expand one slot | 37 | 28/51 | 5/102 | 16,286 |
| Confirmation | Explain all | 39 | 22/51 | 7/102 | 38,116 |
| Extended | Three per slot | 44 | 38/72 | 4/144 | 16,118 |
| Extended | All per slot | 52 | 39/72 | 8/144 | 27,660 |
| Extended | Expand one slot | 51 | 39/72 | 6/144 | 21,850 |
| Extended | Explain all | 52 | 35/72 | 9/144 | 50,956 |

Expanding one clause uses about 21% fewer logical tokens than expanding
both. It loses one strict proof and adds one adverse output in
confirmation, then ties on proofs and emits two fewer adverse
outputs in extended. The three-role menu still uses fewer tokens and
has fewer observed errors in extended. There is no consistent quality
and risk improvement over both references.

With the complete role menu, ten confirmation and thirteen extended
keyword positives have an exact source pair but the raw reader does
not recover both intended roles. Removing the frozen margin cutoff would recover no
additional exact keyword proof from these cases. The three-role menu
therefore leaves some proofs unreachable, while full menu coverage
still leaves substantial recognition error. Adding short definitions
raises token use and lowers strict proof counts in both panels with
this prompt format.

On excerpts, expanding one slot yields 36/51 and 54/72 strict proofs
relative to 38/51 and 53/72 with both expanded. All menus yield zero
keyword substitutions when a source is withheld. Excerpts still
produce substitutions. Keep the full results separated by clue and
source availability.

The shared two-row forward pads to the longer prompt. Logical token
reductions do not measure GPU work, tail latency, document I/O or
end-to-end latency. These experiments also keep the source search
fixed, so the results measure the role reader within the current
search framework.

## Reproduce

Reconstruct the two public projections from the archived manifests.
The restore command verifies the upstream Git object and SHA-256 of
every admitted XML file, then verifies the full projection digest.
For the inference step, restore both frozen Qwen checkpoints and the
pinned NISO example snapshot from the
[reference experiment](../elife-role-feedback-reads-v1#reproduce).
Run each panel in a Slurm CUDA allocation. Repeat with `--reverse`.

```sh
PYTHONPATH=src:. python -m experiments.credit.restore_elife_projection MANIFEST PROJECTION
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.elife_fixed_pair_role_gap PROJECTION MANIFEST QWEN_EMBEDDING \
  QWEN_4B NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/paired-role-verification-v1/results.json NEW_RESULT --menu-ablation
python benchmarks/elife-fixed-pair-role-menu-v1/aggregate.py \
  benchmarks/elife-role-feedback-reads-v1/results.json NEW_AGGREGATE \
  CONFIRMATION_FORWARD CONFIRMATION_REVERSE EXTENDED_FORWARD EXTENDED_REVERSE
```

Slurm tasks 46609_0 through 46609_3 completed on ws3 RTX 5090.
Both traversal orders reproduce all outcomes. Existing route results
match the archived reference. [All aggregates and model provenance](results.json)
