# Locate qualification drift in selective parser rechecks

The [earlier sparse recheck](../enterprise-sparse-clause-recheck-v1)
saved work but changed qualification decisions in selected rows.
This diagnostic keeps the frozen BF16/FP16 block trigger and the
same private 80 development requests and 72
[public heading controls](../enterprise-heading-roles-v1).
It asks whether drift arises from omitted rows, the compact
backbone forward, or the terminal four-label projection.

For each triggered block, preserve the canonical pad width, mask,
and position IDs. Mark the exact subset that disagrees in its
qualification label across precisions or has an FP16 qualification
margin at most 0.5. Compare:

| Recheck | Backbone input | Terminal readout |
| --- | --- | --- |
| Full | All original rows | Original head shape |
| Compact | Only marked rows | Only marked rows |
| Head shape restored | Only marked rows | Place their last hidden vectors in the original head shape and take their original row results |
| Four-row shape | Add original rows to reach four if available | Read the marked rows |
| Eight-row shape | Add original rows to reach eight if available | Read the marked rows |
| Guarded head shape | Compact backbone and both readout shapes | Return to the whole block if either qualification decision disagrees between the two readout shapes or the restored head is within 0.5 logits of the qualification boundary |

The eight-row shape restores the entire backbone block, because
the baseline cap is eight. It keeps the cached labels for
unselected rows and isolates the effect of omission. The other
partial routes also keep their cached unselected labels, except
when the guard returns a full block. The guard is a declared
heuristic, with no threshold search on these cohorts.

For a complementary orientation check, feed canonical full
backbone vectors into the compact head. Compare the final hidden
state per selected row and report last-token activations at
layer boundaries on any row with a changed qualification mark.
On those same rows, compare 13 first-layer probe points across
normalization, query/key/value projections, attention output, and
the projections of the feedforward block. The probes distinguish
an observed activation boundary from the underlying kernel choice.

Measure synchronized fallback kernel calls and count backbone
tensor slots. Blocks and input tensors are prepared before
timing. Parser overhead, the common cached branch, retrieval,
qualification verification, and request transport are outside
those timings. Compare final request-level qualification sets
and public target labels separately. No independent human
enterprise qualification labels enter the diagnostic.

## Results

The 80 development requests trigger nine rechecks. Each block has
eight rows and one selected row. Both traversal orders reproduce
the same decision and activation differences.

| Recheck | Executed tensor slots | Changed selected qualification marks | Changed omitted qualification marks | Requests with changed qualification set |
| --- | ---: | ---: | ---: | ---: |
| Full | 95,032 | 0 | 0 | 0 |
| Compact | 11,879 | 3 | 0 | 2 |
| Head shape restored | 11,879 | 3 | 0 | 2 |
| Four-row shape | 47,516 | 2 | 0 | 2 |
| Eight-row shape | 95,032 | 0 | 0 | 0 |
| Guarded head shape | 42,391 | 3 | 0 | 2 |

Eight-row shape recovers the full qualification sets, and two
other four-label decisions differ because the arm retains cached
labels on omitted rows. The guard invokes four full native
rechecks and still leaves three changed qualification marks.
The dual-precision selection threshold and the narrow score
guard are insufficient for equivalence here.

**The drift is already in the first attention projections.**
Canonical hidden vectors scored by the compact terminal head
produce no new qualification decisions. In contrast, the
selected-row backbone changes all nine chosen hidden vectors.
The largest final logit difference is 1.25. For the three flipped
rows, first-layer input normalization is bitwise identical and
the key and value projections already differ. The query
projection differs for one. These differences occur before the
attention output projection and feedforward computations.
Restoring terminal geometry cannot remove the upstream drift.
The probes identify where tensors differ under these wrappers,
without isolating the specific GPU kernel or accumulation order.

On the nine rechecked blocks, synchronized component P95 is
1.706–1.718 s for the full block and 201.9–202.5 ms for the
compact recheck. Those timings cover only the fallback
computations and give no complete-request estimate. None of the
72 public controls triggers a full recheck under this configuration;
all arms score 61/72 correct. This panel supplies no public
validation of sparse rechecking.

Keep the whole-block reference for exact qualification semantics.
A subsequent optimization should first control the QKV projection
geometry and then test on public controls that invoke the fallback.
Both ws6 RTX 4090 runs in array 47267 agree on all diagnosis
counts, and they reproduce the initial 47260 geometry counts.
[Aggregate results](results.json).

## Reproduce

On a Slurm GPU allocation, use the
[pinned enterprise projections and parser checkpoint](../enterprise-hybrid-rescue-v1):

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python -m experiments.retrieval.enterprise_fallback_batch_geometry \
 DEVELOPMENT_QUERIES benchmarks/enterprise-heading-roles-v1/controls.json \
 QWEN_4B_CHECKPOINT benchmarks/enterprise-hybrid-rescue-v1/results.json NEW_AGGREGATE
```

Repeat with `--reverse` and a new output. Only aggregate evidence
is published.
