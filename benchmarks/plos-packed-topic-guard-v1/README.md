# Guard the ordered expert results of packed topic verification

Use the original top-eight topic candidates and the same author role
declarations in every arm. A declaration shown to be complete can rule
out an article that has no contributor with both required roles. Leave
articles with missing declarations in the verifier batch. The evidence
formatter still reads actual publisher declarations before emitting
all recommendations. In this replay these declarations are available
locally even when masked in the nomination index.

Compare the methods below on the existing 46-query, 23-topic PLOS
replay, with both declaration masks. Use the original cutoff and five
ordered expert–source witnesses in every arm. Calibration uses only
the original pilot and measures the maximum absolute canonical-to-packed
margin difference on retained rows. The packed model scores at the final
token of each independent causal sequence. Its attention offsets isolate
sequences, and the packed and canonical backbones share parameters.

| Arm | Execution and fallback |
| --- | --- |
| Base | Pad and verify all eight source rows. |
| Fixed width | Verify retained rows with the base padding width. |
| Packed | Verify retained rows together with independent variable-length attention. |
| Broad guard | Use packed scores, then rescore the full original batch if a retained score is near the cutoff or adjacent accepting scores are close. |
| Output guard | Use the same packed batch and full rescore, checking only output-bearing sources through the fifth distinct expert or the first safely rejected source. |

With score uncertainty of at most epsilon per source, a margin of more
than epsilon protects a threshold decision. A gap of more than twice
epsilon protects the order of adjacent sources. The output guard
checks the accepted prefix through the last emitting source, including
its separation from the next source. When there are fewer than five
experts, it also checks that the first rejected source safely falls
below the cutoff. Sources with no contributors who meet both roles
cannot affect the output. The interval is an empirical calibration
envelope, so report any test score outside it and all changed citation
lists. A discrete stress check covered 567,060 score perturbations on
generated guard-protected inputs.

Tokenization is shared over the original eight sources, including for
the packed arm, so a full dense fallback can reuse the exact tokens.
Query timing includes topical retrieval, that shared tokenization,
verification, fallback and witness selection. It excludes concurrent
request load, service and external I/O.

## Results

Slurm job `43540` completed on an RTX 5090. The 1,123 calibration
scores give an envelope of 0.4375. The largest raw packed error in the
replay is 0.25 at both masks. The packed model shares all 1,191,553,024
bytes of canonical backbone parameters and requires no duplicate
backbone parameters. Final head weights for the two labels are copied.

| Indexed declarations | Arm | Ordered outputs changed / 46 | Full rechecks | Local P95 |
| --- | --- | ---: | ---: | ---: |
| 42 | Base | 0 | 0 | 65.83 ms |
| 42 | Fixed width | 0 | 0 | 56.63 ms |
| 42 | Packed | 1 | 0 | 36.87 ms |
| 42 | Broad guard | 0 | 28 | 86.49 ms |
| 42 | Output guard | 0 | 19 | 86.06 ms |
| 78 | Base | 0 | 0 | 65.08 ms |
| 78 | Fixed width | 0 | 0 | 48.39 ms |
| 78 | Packed | 1 | 0 | 35.38 ms |
| 78 | Broad guard | 0 | 19 | 81.86 ms |
| 78 | Output guard | 0 | 19 | 81.29 ms |

All arms yield 49 tagged grounded and 14 off-label slots under both
masks. Raw packing reduces P95 against fixed width by 19.76 and
13.01 ms (descriptive paired topic-bootstrap 95% intervals −24.30
to −15.37 and −21.37 to −11.26 ms). The guarded route preserves
ordered results, but reducing whole-batch fallbacks from 28 to 19 at
partial coverage still leaves many requests on the slow path. Fixed
width remains the strongest observed same-output operating point.
The baseline and the packed kernels use the same model and
tokenizations, with different execution arithmetic. Timing and
observed output equivalence refer to these queries on this GPU.

[Aggregate results](results.json)

## Reproduce

Use the pinned projections from the
[topic holdout](../plos-topic-role-holdout-v1) and the CUDA variable-length
attention backend used in the [CoF packed study](../cof-packed-evidence-rescue-v1).

```sh
PYTHONPATH=src:. python -m experiments.credit.packed_topic_guard \
  PILOT_PROJECTION HOLDOUT_PROJECTION HOLDOUT_MANIFEST QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
