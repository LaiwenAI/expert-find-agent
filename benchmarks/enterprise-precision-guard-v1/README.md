# FP16 prefix caching for qualification decisions

The [first-query qualification guard](../enterprise-qualification-guard-v1/README.md) rechecks whole clause blocks when the cached role scores put a clause close to the qualification boundary. This protects the qualification set consumed by expert verification. The rechecks account for much of the parser's tail latency.

Compare BF16 and FP16 with the same pinned Qwen3-4B model, request text, anchored clause prompts, token budgets and block order. The FP16 model is a runtime cast of the checkpoint. At each precision, compare full-context independent scoring and prefix caching. Also compare two cached guard paths. The existing BF16 guard uses a 2.0-logit boundary, while the FP16 guard uses a fixed 0.5-logit boundary. Both fall back to independent BF16 scoring of the original block. These are empirical thresholds. Neither one supplies a bound on qualification errors.

Report downstream qualification additions and removals against independent BF16 scoring. Also count four-way role changes and differences against independent scoring at the same precision. The 72 public controls evaluate the anchored target role. The 80 enterprise development requests have no independent role judgments in this experiment. The timing begins at shared prompt construction and includes cache setup, forward passes, fallbacks and result extraction. Each arm runs separately with rotating order.

## Results on 80 enterprise development requests

| Route | Qualification changes vs independent BF16 | Qualification changes vs independent same precision | BF16 fallback blocks | Parser P95 |
| --- | ---: | ---: | ---: | ---: |
| BF16 independent | 0 | 0 | 0 | 5,258 ms |
| BF16 cache | 3 | 3 | 0 | 439 ms |
| BF16 guard | 0 | 0 | 25 | 1,846 ms |
| FP16 independent | 3 | 0 | 0 | 5,354 ms |
| FP16 cache | 3 | 0 | 0 | 445 ms |
| FP16 guard | 1 | 2 | 7 | 1,014 ms |

FP16 caching reproduces all qualification decisions of independent FP16 scoring. Its maximum absolute logit drift from its own reference is 0.375, compared with 2.5 for the BF16 cache and its reference. Converting the independent path from BF16 to FP16 itself changes three qualification decisions. The FP16 guard removes one qualification relative to independent BF16 and lowers observed parser P95 by 45.0% relative to the BF16 guard. The BF16-over-FP16-guard P95 ratio has a paired request-bootstrap interval of 1.01–3.73×. The BF16 guard alone preserves the independent BF16 qualification set among the cached guard paths compared here.

All six arms score 60 of 72 public target roles correctly, with two false qualifications and ten missed qualifications. The targets come from 36 constructed prior-qualification and future-work pairs. These controls do not resolve the role discrepancy in the private panel. Parser latency does not include candidate retrieval, verification or service overhead.

Job 41416 ran the experiment on an RTX 5090 on ws5. Only aggregate counts and model/data hashes are released. [Aggregate results](results.json)

## Reproduce

Use the same development-only projection as the [first-query comparison](../enterprise-first-query-v1/README.md) and the [public controls](../enterprise-heading-roles-v1/controls.json). Run in a Slurm CUDA allocation with sufficient capacity to hold both model instances and a new output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_precision_guard \
  DEVELOPMENT_QUERIES PUBLIC_CONTROLS QWEN_4B_CHECKPOINT NEW_RESULTS
```
