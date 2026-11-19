# Continue cached reviewer evidence or choose the reading depth upfront

Keep the fixed title-group splits, Qwen3-4B-Instruct-2507 checkpoint, and instruction from the [MERIT evidence-dispatch study](../merit-evidence-dispatch-v1/README.md). The first-pass input reads the first two publications. The full input uses the original profile order. Each test arm runs independently, with rotating arm order.

| Route | Evidence computation |
| --- | --- |
| First two | Score two publications once |
| Full | Score the full profile once |
| Serial rerun | Score two; on small prediction margin, rescore the complete prompt |
| Cached continuation | Use the same trigger, crop the cache to the exact token prefix shared by both prompts, and evaluate only the remaining full-prompt tokens |
| Guarded continuation | Use cached continuation and repeat the full pass if its continued margin is within 0.5 logits of zero |
| Upfront depth | Estimate the benefit of a full read from pre-model term coverage and profile length, then score either two publications or the entire history once |

The learned gate fits improvement in correctness on the fit partition. Calibration uses up to 25% of the full-route token total for additional reading. For serial rerun, it chooses the most inclusive margin threshold within the budget. For upfront depth, it maximizes balanced accuracy within the same allowance, charging full-minus-short tokens because it makes one forward call. Both thresholds remain frozen on test. Upfront features read already resident profile text for lexical coverage, and timing includes that pass. The token budget is not a wall-clock guarantee.

The continuation routes compute the prefix from actual token IDs after full tokenization when fallback is required. Cache cropping discards closing prompt markers and any boundary tokens that differ between the short and full views. Token counts reflect rows actually passed through the model, including repeated work when an arm rechecks. Serial warm timing includes prompt assembly, tokenization, GPU calls, cache maintenance, and gate computation. Evaluation uses reviewer self-assessment and does not label individual references.

Prefix caching and variable-depth context selection are established techniques. This run tests their use within a reviewer evidence decision under a common checkpoint. [Prompt Cache, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/a66caa1703fe34705a4368c3014c1966-Abstract-Conference.html) | [Adaptive-k, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1017/)

## Results on RTX 5090, 245 test pairs

| Route | Accuracy | Balanced accuracy | Positive F1 | P95 | Decisions differing from serial rerun | Requests reading full profile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First two | 70.61% | 70.65% | 65.38% | 148.7 ms | 17 | 0 |
| Full | 68.16% | 71.23% | 67.77% | 237.4 ms | 35 | 245 |
| Serial rerun | 66.94% | 68.00% | 63.35% | 296.8 ms | 0 | 39 |
| Cached continuation | 67.76% | 68.67% | 63.93% | 203.6 ms | 2 | 39 |
| Guarded continuation | 67.35% | 68.34% | 63.64% | 205.9 ms | 1 | 39 |
| Upfront depth | 70.61% | 71.95% | 67.57% | 237.8 ms | 31 | 113 |

All continuations took the same 39 escalation decisions as the serial rerun. Cached continuation lowered P95 by 31.4% and mean time among escalated requests from 272.3 to 186.9 ms. Two of 245 final labels changed. The 0.5-logit guard repeated one full pass and left one label mismatch. Exact token-prefix identity therefore does not ensure decision parity in BF16.

Upfront depth matches the accuracy count of fixed two and increases positive F1 from 65.38% to 67.57%, with P95 near the full-profile control. Its paired accuracy difference against full reading is +2.45 percentage points (95% paper-cluster interval -1.27 to +6.17). Fixed two remains the latency control. These are single-GPU reviewer self-assessment results, with no individual citation judgments. [Aggregates](results.json)

## Reproduce

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.decision.merit_prefix_dispatch \
  MERIT_ASSESSOR_TEST_JSON QWEN_4B_CHECKPOINT NEW_RESULTS
```
