# Test compact qualification rechecks with projection controls

The [batch geometry diagnostic](../enterprise-fallback-batch-geometry-v1)
found that compact BF16 rechecks changed qualification decisions
and that activation drift was already visible in the first
attention projections. This study tests interventions at that
boundary. Batch-size sensitivity is also documented for language
model generation by [Yuan et al. (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f80094a824ba5912d4a2de169c404a40-Abstract-Conference.html).
Here the outcome is a classification of the original request
clause as a mandatory prior qualification or another role.

## Design

Keep the Qwen3-4B reader, calibrated BF16/FP16 trigger, and
padded sequence width, attention mask, and position IDs from
the full-block recheck. Compare these controls only during BF16
recheck:

| Arm | Input and computation |
| --- | --- |
| Full | All rows in the native block |
| Compact | Marked rows only |
| Reduction off | Compact with BF16 reduced-precision matmul reductions disabled and the default split-K setting preserved |
| First QKV | Compact, with original batch geometry restored only for the first layer's Q, K, and V projections |
| All QKV | Compact, with original batch geometry restored for Q, K, and V in every layer |

For each restored projection, scatter compact rows to their
original indices in a zero-filled carrier, apply the linear
layer, and gather those rows. The full-shaped GEMMs in these
arms still incur projection work. Compact input slots count
only the backbone input and cannot measure the cost of these
GEMMs. Use synchronized recheck timings for the component
comparison. Timing includes projection scatter and gather
and excludes common gate work, input preparation, and search.

On the 80 private development requests, preserve cached
labels for omitted rows in all compact arms. Compare each
request's qualification set with the reference. The reference
is a model output. Human judgments are available only for
the annotated public target clauses. To build a longer public
stress case, append fixed administrative context to each of 36
English heading controls and preserve the literal target and
its labeled local context. Score the ordinary gated parse. In
a separate intervention, force a selected-row recheck of each
public target using its ordinary block to probe numerical
stability even when the calibrated gate would skip the fallback.
The constructed appendix and forced selection serve as controls
and cannot estimate natural trigger prevalence.

The installed CUDA backend rejected disabling split-K without
changing the matmul implementation to cuBLASLt. The reported
reduction arm preserves split-K and tests the reduced-precision
flag independently of a backend change. See the
[PyTorch CUDA documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html).

## Results

The two traversal orders on RTX 4090s (Slurm array 47280 on ws6,
PyTorch 2.15.0+cu132) reproduce all decision counts and measured
numerical differences. On the private requests the natural gate
rechecks nine eight-row blocks with one selected row each.

| Arm | Changed selected qualification marks / 9 | Changed request qualification sets / 80 | Private fallback P95 (ms) | Changed public forced target labels / 36 | Correct public forced targets / 36 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 0 | 0 | 1,700 | 0 | 25 |
| Compact | 3 | 2 | 200 | 0 | 25 |
| Reduction off | 4 | 3 | 205–207 | 1 | 26 |
| First QKV | 4 | 3 | 204 | 1 | 26 |
| All QKV | 5 | 4 | 324 | 0 | 25 |

All arms agree on the 36 target labels in ordinary public parsing
(25 correct, 11 missed qualifications and zero false marks).
There is one naturally triggered block among the expanded public
cases. No arm changes its qualification set. The forced public
run gives each arm 36 controlled rechecks. Disabling reduced-
precision reductions and restoring first-layer QKV each correct
one target relative to the full-block reference. They also change
more selected qualification marks on private requests than
unmodified compaction. Accordingly, full-block equality is an
engineering stability check and public gold labels are a separate
measure of classification quality.

Restoring QKV geometry at every layer increases component cost
without preserving private qualification sets. The earlier QKV
differences identify an early point of drift, and these interventions
show that matching QKV geometry alone is insufficient. Keep
whole-block rechecks as the reference. Future qualification changes
need independently judged clauses and end-to-end timing, including
fallback. The private output changes have no human judgment, and
the reported P95 covers nine component rechecks per run.

[Aggregate runs and output digests](results.json)

## Reproduce

Restore the private 80-query development projection from the
[hybrid benchmark](../enterprise-hybrid-rescue-v1), and use the
same Qwen3-4B checkpoint on an allocated Slurm CUDA node. Repeat
with `--reverse` and a fresh output path.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_qkv_stability \
  DEVELOPMENT_QUERIES benchmarks/enterprise-heading-roles-v1/controls.json \
  QWEN3_4B_CHECKPOINT benchmarks/enterprise-hybrid-rescue-v1/results.json NEW_RESULTS.json
```
