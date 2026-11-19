# Calibrated adaptive reranking depth on QASPER

Compare fixed reranking with a learned choice of dense-only or 8, 16 or 32 reranker checks. An eight-feature ridge model predicts complete evidence for each action using the paragraph count and dense scores, including the gap around fifth place and gaps to deeper positions. The frozen checkpoints and fusion rule come from the [certified reranking diagnostic](../qasper-certified-reranking-v1/README.md).

Take one hash-selected question per official development paper. Hash paper IDs into separate fitting and calibration sets. Fit only on alignable evidence annotations. Calibrate two policies with check budgets equal to the actual mean checks of fixed 16 and fixed 32 in the calibration set. For each budget, select the cost penalty that maximizes complete evidence among candidates meeting the budget. Count checks over all requests, including those without alignable annotations.

Apply the selected policies to the same 96 official test papers as the earlier diagnostic. Every live route starts from the query text and independently encodes it, scores the known paper's paragraphs and runs its chosen reranker blocks. The timing includes routing and tokenization. Test labels supply only evaluation. The earlier study already used the test panel. Document discovery, serving and expert attribution are outside this experiment.

## Results

| Test route | Mean checks | Complete evidence (83 evaluable) | Evidence F1 | Request P95 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.00 | 55.42% | 26.49% | 7.06 ms |
| Fixed 8 | 8.00 | 61.45% | 29.14% | 42.70 ms |
| Fixed 16 | 15.93 | 68.67% | 31.72% | 74.77 ms |
| Fixed 32 | 30.01 | 69.88% | 32.07% | 132.36 ms |
| Adaptive, 16-check budget | 14.34 | 68.67% | 31.54% | 108.39 ms |
| Adaptive, 32-check budget | 14.34 | 68.67% | 31.54% | 108.76 ms |
| Certified | 38.50 | 69.88% | 32.07% | 235.99 ms |
| Exhaustive | 52.65 | 69.88% | 32.07% | 294.33 ms |

Fit on 205 development papers (173 evaluable) and calibrate on 76 (61 evaluable). Both budgets selected the same policy. On test it chose dense-only 23 times, depth 8 once, depth 16 on 58 queries and depth 32 on 14. The 16-check policy saved about 10% of average checks and raised P95 by 45.0% relative to fixed 16. Its complete-evidence difference was 0 percentage points (descriptive paired interval -4.82 to +4.82). The policy also lagged fixed 16 on the calibration quality metric (62.30% vs 63.93%). Keep fixed 16 as the faster reference under the measured quality criteria. Fixed, certified and exhaustive output order reproduced the prior GPU run on all 96 test queries.

As a hindsight bound on the 83 evidence-evaluable requests, give each request its cheapest successful action in the four-action set. Within either fixed route's total checks, this recovers at most 59 complete evidence sets, compared with 57 for fixed 16 and 58 for fixed 32. The oracle uses test annotations and bounds this prefix action set and quality metric only. Gains from better source selection or richer actions remain to be measured.

## Reproduce

On a Slurm GPU compute node with the [official QASPER data](https://aclanthology.org/2021.naacl-main.365/) and the pinned checkpoints:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.qasper.adaptive_reranking_depth \
  DATA_DIRECTORY EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON
```

[Aggregate results](results.json) contain the calibration grid, frozen router coefficients and model and input hashes. Raw per-question results and text stay local.
