# Support verification under explicit contradiction

Use the [official SciFact corpus](https://aclanthology.org/2020.emnlp-main.609/) to compare the enterprise verifier's current relevance-style prompt against direct support verification. Hold the supplied abstract fixed for each case. Include complete abstracts of up to 2,500 characters. Support and contradiction come from annotated claim–abstract pairs. The unknown diagnostic pairs claims with no released evidence with their original cited abstracts. This last class is benchmark-derived. A fixed hash selects at most one case per claim and per paper within each panel.

Calibrate on 309 pairs from the official training split after excluding every paper referenced by a development claim. Evaluate on 226 pairs from the official development split. Claim IDs and papers are disjoint across these two constructed panels. The public corpus has been used in earlier exploratory experiments. This test evaluates whether a supplied abstract supports a fact. Expert selection and personal contribution are outside its scope.

The 0.6B reranker compares the original enterprise instruction, an explicit claim-support instruction, and a score subtracting any positive contradiction margin from the explicit support margin. For each calibrated route, select a frozen threshold that retains as many true supports as possible subject to separate 5% false-support limits for contradiction and unknown in calibration. The direct reader uses the open Qwen3-4B-Instruct-2507 checkpoint and one fixed constrained digit mapping for support, contradiction, and unknown. Its calibrated variant thresholds the support logit minus the larger of the competing logits.

## RTX 5090 results

| Route | Accepted annotated supports / 97 | Accepted contradictions / 47 | Accepted unknowns / 82 | P95 per batch of up to 16 |
| --- | ---: | ---: | ---: | ---: |
| Original reranker, zero cutoff | 91 | 41 | 37 | 154.2 ms |
| Explicit support reranker, zero cutoff | 89 | 36 | 19 | 156.6 ms |
| Original reranker, calibrated | 5 | 1 | 1 | 154.2 ms |
| Explicit support reranker, calibrated | 12 | 3 | 1 | 156.6 ms |
| Contradiction-adjusted reranker, calibrated | 1 | 4 | 0 | Two calls |
| Direct three-way reader | 49 | 4 | 2 | 798.2 ms |
| Direct reader, calibrated | 48 | 4 | 2 | 798.2 ms |

Explicit support wording reduces false support, yet the uncalibrated reranker still accepts 36 of 47 contradictions. The stricter cutoff largely removes support coverage. The direct reader accepts many fewer negatives and recovers more positives than the calibrated reranker. It misses nearly half of the true supports and costs more compute. The direct reader predicts contradiction for 38 of the 97 true supports, so label polarity and mapping warrant separate checks. Calibration limits do not transfer perfectly to development.

Timing measures tokenization and inference on identical batches of up to sixteen. The direct reader and reranker were run in separate jobs on the same compute node. Model loading, search, and service transport are outside the timed batch. [Reranker results](reranker.json) · [Reader results](reader.json)

## Reproduce

Use the official archive with the recorded digest and the pinned Qwen checkpoints:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.criterion_verifier_calibration \
  --archive-path SCIFACT_ARCHIVE --checkpoint QWEN_RERANKER --output NEW_RERANKER_RESULTS
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.criterion_reader \
  --archive-path SCIFACT_ARCHIVE --checkpoint QWEN_4B_READER --output NEW_READER_RESULTS
```

Jobs 41158 and 41166 completed on ws5. An optional local diagnostic prioritizes model disagreements for blinded enterprise review. Its cases and outputs stay private and have no human judgments.
