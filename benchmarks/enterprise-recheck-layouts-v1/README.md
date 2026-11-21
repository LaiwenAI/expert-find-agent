# Recheck each clause against the full request with independent layouts

The [first-query phase timings](../enterprise-prefix-fork-v1) put most
of the observed tail latency in qualification parsing. Replacing an
entire independent BF16 fallback with a shared-prefix packed forward
previously changed qualification membership on two requests. The
previous guard replayed all seven canonical blocks.
[Packed shared-prefix comparison](../enterprise-packed-clause-recheck-v1)

Keep the original BF16/FP16 cached parser and fallback trigger.
Each arm sees the same complete input tokens for every clause of a
triggered block. Only the independent full-context BF16 layout changes. Ragged attention
is established serving infrastructure. The question here is whether
changing batch layout preserves the qualification decisions that control
this particular retrieval pipeline.
[FlashInfer, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html)

| Arm | Recheck |
| --- | --- |
| Whole block | Perform the original left-padded BF16 forward across all clauses in one block. |
| Variable length | Preserve independent full sequences with causal attention and per-sequence positions. Remove padding using GPU variable-length attention. Share the reference model weights. |
| Four rows | Run each canonical block in contiguous groups of at most four full clause rows. |
| Agreement | Try variable-length and four-row layouts. Keep variable-length logits if qualification decisions agree and both absolute qualification margins exceed 0.5 everywhere. Otherwise replay the original whole-block forward. |

Check sequence isolation on the GPU. Compare all four role labels,
the mandatory qualification spans consumed by the search, retrieved
experts, issued people, fallback decisions and exact citation offsets.
The live search shares its frozen parser weights, hybrid-precision
trigger, source index, embedding and guarded packed verifier across arms.
Each run processes the same 80 original development requests and 72
existing bilingual synthetic controls. Arm order rotates per request
and is reversed in the replica. Warm serial timing excludes loading,
index construction and service queues. The paired intervals resample
development requests and are descriptive for this cohort.

In a separate stress test, select two controls per original language,
layout and target-role stratum, yielding 24 distinct controls. Prepend
one or three copies of unrelated archival background in the same
language. Preserve each original anchored target. Force the canonical
block containing that target through each of the four layouts. This
test measures full-context layout agreement and block latency. A third
condition uses eight copies, approximating the average padded width of
triggered enterprise blocks. It also runs the inherited cached trigger
for the selected target block and reports the number that naturally
requires rechecking. These are synthetic controls for numerical
sensitivity. Enterprise matching quality requires independent labels.

## Live results

Both execution orders produce the same non-timing signature. Seven
blocks across six development requests reach mandatory qualification
rechecks.

| Recheck | Warm first-query P95 (ms) | Requests with changed qualifications | Requests with a recommendation / 80 | BF16 fallback token slots |
| --- | ---: | ---: | ---: | ---: |
| Whole block | 1062–1069 | 0 | 37 | 74,680 |
| Variable length | 1055–1062 | 4 | 36 | 74,122 |
| Four rows | 1061–1067 | 2 | 36 | 74,352 |
| Agreement | 1549–1568 | 0 | 37 | 223,154 |

The two alternative layouts change one recovery recommendation each.
Their initial recommendations and retrieval order stay fixed. Across
the six recheck requests, variable length reduces median complete
latency by about 115 ms in each run. Across all 80 requests, the P95
point estimate improves by approximately 7 ms.
The paired descriptive interval touches or nearly touches zero on the
two runs. Agreement sends all seven blocks to the original path and
keeps all original search outputs. All four methods label 60 of 72
anchored public controls correctly under the natural trigger, although
three methods change at least one full label sequence. The original
whole block remains the search reference. These comparisons measure
output drift, with no adjudicated quality label for the changed
enterprise recommendations.

## Forced-block controls

The shorter synthetic groups use prefix lengths of 373–426 and
629–683 tokens. All four rechecks preserve every categorical label in
both groups. On the longer of the two, variable-length block P95 is
297–303 ms versus 330–337 ms for whole-block execution. These groups
are shorter than the enterprise trigger geometry. The 56 enterprise
rows use an average of 1,334 padded tokens.

The length-matched group uses 1,259–1,333 prefix tokens. Only one of
24 target blocks meets the original trigger despite the longer input.
All layouts agree on all 24 blocks, including the naturally triggered
one. Variable-length block P95 is about 15% lower in this forced
comparison. The agreement guard rejects once and nearly doubles
P95 because it runs both layouts. Matching input length alone fails
to reproduce the enterprise decision-boundary sensitivity. The live
replays remain the appropriate evidence for choosing a default.

## Reproduce

Restore the restricted source projection, exact vectors, and checkpoints
from the [guarded packed recovery experiment](../enterprise-packed-role-search-v1).
Use a Slurm GPU allocation, pinned controls and separate outputs.
Repeat with `--reverse` for the other execution order.

```sh
export PYTHONPATH=src:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.enterprise_recheck_layouts \
  QUERIES CORPUS ROLE_WINDOWS VECTORS META MODELS PACKED_REFERENCE CONTROLS RESULT
python -m experiments.retrieval.enterprise_recheck_stress CONTROLS MODELS STRESS_RESULT
python -m experiments.retrieval.enterprise_recheck_stress CONTROLS MODELS MATCHED_RESULT --length-matched
```

The [aggregate results](results.json) record model and input provenance
and digests of the ordered non-timing outputs. No per-request private
data or candidate identities are included.
