# Cost-aware evidence reading on MERIT reviewer ratings

Use all 610 released MERIT assessor pairs. A reviewer rating of 4 or 5 is positive, and 1 to 3 is negative. Exact normalized target-paper titles group pairs. A frozen title-hash rule draws 231 fit pairs, 134 calibration pairs, and 245 test pairs. The overall public corpus has appeared in prior aggregate studies. These splits isolate learning of the new controller. Reviewer identities are unavailable for grouping. [MERIT, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.2227/)

Every arm uses the same frozen Qwen3-4B-Instruct-2507 and a constrained single-digit decision from the final-token logits. The model sees the full target title, abstract, and introduction. The compact controls read two reviewer publications, using the first two, the two largest lexical overlaps, or one lexical match followed by a source covering residual target terms. Lexical selection uses the target title and abstract. The full control reads all supplied publications. No target rating is passed to either the model or source selection.

Both controllers first use complementary selection. The margin control favors small absolute decision margins. The learned controller regresses the change in correctness from reading the full history using the margin, term coverage, residual coverage, and profile features. It divides predicted gain by the length of the full record in characters. Thresholds are selected on the calibration split to maximize correct decisions while using at most 40% of the full-route tokens as extra work. This is a calibration budget, not a hard test-time cap. An escalation executes a fresh full forward pass with no prefix reuse.

Warm serial measurements include source selection, tokenization, both passes when applicable, and the gate. Timing excludes model loading and network transport. The benchmark measures agreement with reviewer self-assessment. It supplies neither citation-level reference labels nor enterprise suitability judgments.

## RTX 5090 results on 245 test pairs

| Route | Agreement | Positive-class F1 | Mean latency | P95 | Full-history fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: |
| First two | 70.61% | 65.38% | 106.4 ms | 149.0 ms | 0 |
| Top two by overlap | 69.39% | 65.75% | 107.3 ms | 147.6 ms | 0 |
| Complementary two | 67.76% | 62.20% | 107.3 ms | 147.7 ms | 0 |
| Full history | 68.16% | 67.77% | 152.0 ms | 237.3 ms | 0 |
| Margin fallback | 67.35% | 63.64% | 127.6 ms | 266.8 ms | 34 |
| Learned fallback | 66.94% | 60.87% | 122.1 ms | 241.6 ms | 29 |

The first-two route reduces P95 by 37.2% relative to full history, with an observed accuracy difference of +2.45 percentage points (95% paper-cluster bootstrap interval -3.29 to +8.19). Full history has the highest reported positive-class F1. The learned policy escalates less often than margin routing, and both reduce mean time versus full history at the cost of higher P95. Neither improves test accuracy over its own compact starting route. All arms finished. These comparisons use one GPU and a previously studied public source cohort. [Aggregates](results.json)

The results motivate a one-pass selection control in addition to a full-history baseline. Recent work studies variable-depth one-pass context selection in QA, a separate task with no expert-level provenance check. [Adaptive-k, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1017/)

## Reproduce

Run in a Slurm CUDA allocation with the pinned MERIT assay and the verified Qwen checkpoint:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.decision.merit_evidence_dispatch \
  MERIT_ASSESSOR_TEST_JSON QWEN_4B_CHECKPOINT NEW_RESULTS
```
