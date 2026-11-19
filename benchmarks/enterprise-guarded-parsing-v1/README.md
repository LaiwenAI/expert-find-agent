# Rechecking ambiguous full-context clause batches

Use the same request, tokenization, individual clause suffixes and frozen Qwen3-4B-Instruct-2507 as the [shared-context benchmark](../enterprise-isolated-parsing-v1/README.md). For the adaptive arm, score each block using the shared prefix cache. If the smallest top-two logit gap in a block is within the frozen band, re-evaluate that entire block using the original independent batched execution. A recheck preserves the standalone batch geometry. This targets label agreement with the independent execution, which is a reference model output.

Select the band from the prespecified grid on the 72 public constructed heading controls. Minimize the number of residual disagreements, then the number of rechecked blocks, then the cutoff. The fixed control rechecks blocks by a stable request-and-block hash at the expected rate from the calibration set. The actual rates on the 80 enterprise development requests can differ. Each execution includes prompt assembly, inference, and any rechecks. Rotate the method order by request. No data from the enterprise evaluation is used in selecting the band.

## RTX 5090 results

The public controls selected a 2.0-logit band, rechecking 18 of 96 calibration blocks to recover both disagreements. All four arms got 60 of 72 public target labels right. The same control set was used for tuning and this check.

| Mode on 80 development requests | Rechecked blocks / 218 | Clause disagreements / 1,452 | Qualification additions / removals | P50 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent full batch | 0 | 0 | 0 / 0 | 503 ms | 5,263 ms |
| Prefix cache | 0 | 16 | 2 / 1 | 118 ms | 439 ms |
| Hash recheck | 46 | 15 | 2 / 0 | 169 ms | 2,171 ms |
| Margin recheck | 56 | 0 | 0 / 0 | 161 ms | 3,287 ms |

The margin-based arm reproduced all reference labels on this development cohort while lowering P95 by 37.5% against the independent full batch. The rechecks made P95 about 7.5 times the cache-only measurement. The hash control used a different realized recheck count and does not establish a fixed-cost advantage. Margins are model scores, and agreement with a reference alone cannot establish semantic accuracy or numerical invariance on unseen requests. [Measured aggregates](results.json)

## Reproduce

Inside a Slurm GPU allocation, use the same input projection and checkpoint as in the linked shared-context benchmark:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_guarded_parsing \
  DEVELOPMENT_JSONL benchmarks/enterprise-heading-roles-v1/controls.json \
  QWEN_CHECKPOINT NEW_AGGREGATE
```

The aggregate includes stable hashes and no request texts. The timed public controls were also used to select the band. Private-development agreement measures model execution stability, with human assessment of qualification decisions still pending. This is parser-stage timing.
