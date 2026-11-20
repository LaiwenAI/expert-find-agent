# Resolve uncertain packed scores with a partial padded recheck

The [gap-refill experiment](../role-gap-title-refill-v1) measured
quality effects at the same four-source reading budget. The earlier
[packed-verifier replay](../plos-packed-topic-guard-v1) found that
variable-length attention lowered raw latency and changed some
citations. Rechecking every row of an uncertain batch erased the
speed advantage. This study keeps the nomination policy fixed and
tests selective numerical rechecks at request time.

| Verifier execution | Behavior |
| --- | --- |
| Padded | Score the four original rows together. |
| Fixed width | Omit structurally excluded rows while retaining original width. |
| Packed | Score the active rows in one variable-length forward with model weights shared with the padded path. |
| Full recheck | Use the calibrated output guard and rescore every original row if the emitted order may change. |
| Selective recheck | On the same guard, rescore only output-bearing candidates whose intervals cross the acceptance threshold or overlap an adjacent possible accepted candidate. Preserve the original padded width. If a new score escapes its interval, rescore the original full batch. |

The uncertainty radius is the prior pilot's frozen value of 0.4375.
Under that radius and exact canonical rechecked scores, a row safely
below the threshold cannot emit. Within the remaining ordered rows,
disjoint neighboring intervals protect their order. Rechecking the
endpoints of overlapping adjacent intervals resolves the remaining
comparisons. The bound is empirical, and partial shape changes can
affect the reference arithmetic. Measure raw packed scores outside
the envelope, unequal rechecked scores and changes to the final
serialized expert–source witnesses separately. A local synthetic
stress check covered 50,625 bounded perturbations with role overlap.

Use the frozen PLOS and eLife queries, complete and partial visibility
masks, candidate refill, embedding and reranker. Reproduce the padded
citation aggregates of the previous study. Record query encoding,
candidate retrieval, shared tokenization, the chosen verifier execution,
and serialization as a warm local serial request. Report execution
token slots, the number of fallback launches and paired tail-latency
intervals. External source I/O, service queues, query planning and
enterprise human adjudication are outside this replay.

## Results

The following table uses full declaration visibility and timing from
job `44028` on ws5. Job `44031` matches all non-timing metrics. Exact responses are byte-identical ordered
expert–source pairs relative to the padded reference.

| Execution | PLOS exact / 46 | PLOS P95 | eLife exact / 110 | eLife P95 |
| --- | ---: | ---: | ---: | ---: |
| Padded | 46 | 35.34 ms | 110 | 29.38 ms |
| Fixed width | 46 | 35.34 ms | 110 | 29.39 ms |
| Packed | 43 | 25.63 ms | 108 | 22.49 ms |
| Full recheck | 46 | 48.00 ms | 110 | 36.82 ms |
| Selective recheck | 43 | 41.33 ms | 108 | 33.56 ms |

Across both visibility masks and both catalogs, the selective
recheck scores 252 rows, versus 504 with full-batch rechecking.
Of its rescored rows, 100 differ numerically from the canonical
four-row score even at the original padding width. All observed
packed scores fall within the frozen envelope, but only 304 of
312 selective outputs agree with the padded reference. The full
recheck reproduces all 312 and increases P95. The selected lists
already contain four eligible rows, so fixed-width compaction has
no rows to skip here.

Preserving padding width alone does not preserve the canonical
reference when batch size changes. The selective route remains an
ablation. A faster first-query verifier needs either a kernel
with validated shape consistency or a serving strategy that avoids
a second launch while maintaining the measured quality criteria.

[Aggregate results](results.json)

## Reproduce

Use the projections from the [gap refill](../role-gap-title-refill-v1)
and a Slurm CUDA environment that supports PyTorch variable-length
attention.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.role_gap_selective_recheck PLOS_PROJECTION ELIFE_PROJECTION \
  benchmarks/elife-mesh-v1/mesh.json benchmarks/role-gap-title-refill-v1/results.json \
  benchmarks/plos-packed-topic-guard-v1/results.json QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
