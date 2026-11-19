# Live witness selection at a calibrated role-acceptance threshold

Use the pinned [PLOS linked-witness fixture](../plos-linked-witness-dispatch-v1/README.md) of 178 matched requests. Supply two paper locations and three scoped role phrases. Encode queries with frozen Qwen3-Embedding-0.6B. Use the same owner-linked candidate records, query vectors, and frozen MiniLM NLI checkpoint in every route. The verifier reads prose formed from each author's publisher-declared roles and NISO definitions. Publisher-declared roles supply the evaluation labels. The live scheduler updates its state from model verdicts after each read. Do not mix declarations between authors in a recommendation.

Assign the seven connected publication components by stable hash, using three for calibration and four for evaluation. The split has 60 calibration and 118 test requests, with no shared publication or publisher owner key. Use only examples from linked candidates for calibration. Deduplicate by article pair, declaration, role, and query wording before choosing fixed thresholds at 10%, 20%, and 30% observed false acceptance on wrong roles. These are atom-level targets on calibration. Test wrong recommendations are counted separately.

Compare these schedules at two and four maximum declaration reads:

| Schedule | Action selection |
| --- | --- |
| Source round robin | Rank declarations independently within each paper, then alternate papers |
| Owner mean | Rank owner plans by average scoped similarity and read the chosen declarations sequentially |
| Adaptive bottleneck | Recompute owner plans after each observed declaration verdict |
| Owner mean batched | Use the same owner-mean read order and score all selected tasks in one batch |

Each sequential route batches all tasks associated with its current declaration. A separate exact-field replay evaluates the same read budgets and uses publisher labels on each selected read to isolate scheduling losses. The fully batched MiniLM arm may change numeric margins. Count matches of both the selected records and final candidate sets against its sequential control. Timed request work includes query encoding, in-memory ranking, rendering on each chosen read and GPU verification. Publication discovery, external reads, offline indexing, parsing, transport, and offline calibration are outside the timing. The requests reuse a small number of paper components and a common wording inventory.

## RTX 5090 results

Job 42366 reproduces the 10% decisions from job 42355. Calibration includes 86 positive and 35 wrong-role checks. The 10%, 20%, and 30% gates accept 24, 37, and 47 positive checks and 3, 7, and 10 wrong-role checks. These counts are from calibration. There are 59 positive and 59 negative test requests.

| Two-read schedule | Positive requests recovered with exact fields / 59 | Positive requests recovered with MiniLM at 10% / 59 | Positive requests recovered with MiniLM at 30% / 59 | Unsupported returns on negative test requests at 30% / 59 |
| --- | ---: | ---: | ---: | ---: |
| Source round robin | 51 | 0 | 3 | 3 |
| Owner mean | 58 | 0 | 3 | 3 |
| Adaptive bottleneck | 53 | 0 | 3 | 3 |
| Owner mean batched | Same read list as owner mean | 0 | 3 | 3 |

At four reads the exact-field replay recovers all 59 positives with 235 adaptive reads, compared with 276 for either fixed ordering. At the 20% gate, MiniLM recovers one positive and promotes no negative requests for every route and both budgets. At 30%, it recovers three and also promotes three negatives. Increasing the read limit from two to four does not improve these results. Source ownership helps disambiguate where to look when role evidence is reliable. In this fixture, the live semantic reader dominates the lost recommendations.

Across every gate and both budgets, the batched arm matches the sequential owner's declaration reads and returned sets on all 118 requests. At the 30% gate and four reads it lowers component P95 from 14.50 to 10.96 ms. Quality at this operating point remains poor. The result measures whether batching is a viable implementation of fixed routing on this workload. It does not establish a quality or end-to-end search advantage over a production service. [Aggregate results](results.json)

## Reproduce

Use the pinned [publisher projection and NISO examples](../plos-linked-witness-dispatch-v1/README.md#reproduce) and verified Qwen embedding checkpoint. Get the MiniLM checkpoint from the [small-reader manifest](../scifact-small-nli-v1/checkpoints.json). Run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.calibrated_witness_search \
    PUBLISHER_PROJECTION NISO_EXAMPLES QWEN_EMBEDDING MINILM_CHECKPOINT NEW_RESULTS
```
