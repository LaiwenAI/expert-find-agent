# Selective cross-precision rechecks in live expert search

Connect the [selective clause reader](../enterprise-cross-precision-v1/README.md) to the same open-model [first-query pipeline](../enterprise-first-query-v1/README.md). Compare independent full-context BF16 scoring, the 2.0-logit BF16 guard, and the selective FP16 cross-check followed by BF16 fallback. The thresholds are inherited, and the selective route uses FP16 only for cached blocks selected by the BF16 2.0 trigger. All arms run the qualification reader, query encoder, exact owned-source index, and source verifier independently. Each returns the same data contract with ordered experts, an evidence witness per supported criterion, and explicit unresolved outcomes.

Use the pinned 80 development requests and 52,667 source records. The experiment rotates all six arm orderings across requests and runs in one CUDA allocation. Each route measures elapsed time from clause extraction to ranked output. Offline indexing, model loading, request transport and concurrency are outside the timed window. Two parser copies remain resident so that the precision check has no model-loading cost during a request.

## Results

| Route | Qualification, rank or citation changes versus independent BF16 | Whole BF16 rechecks | Parser P95 | First-query P50 / P95 |
| --- | ---: | ---: | ---: | ---: |
| Independent BF16 | 0/80 | 0 | 5,272 ms | 571 / 5,475 ms |
| BF16 2.0 guard | 0/80 | 25 | 1,850 ms | 194 / 2,152 ms |
| Selective cross-precision | 0/80 | 7 | 1,001 ms | 195 / 1,184 ms |

The selective path lowers first-query P95 by 45.0% relative to the BF16 guard. Its paired request-bootstrap speedup interval is 1.01–2.72×. Medians are similar. All three paths agree on the qualification spans, candidate order, ranked experts, citations, supported status and unresolved states. Each scores 1,280 source–criterion pairs and returns 70 provisionally supported experts over 34 requests. The selective path changes 12 fine-grained clause labels, with unchanged qualification status.

Job 41472 ran on an RTX 5090 on ws5. This is a repeated development-slice test under warm, serial execution. It establishes agreement with the frozen reference and tail latency for the complete local search path. Independent human relevance and attribution judgments, data held out from development, and timing under service load remain required. [Aggregate results](results.json)

## Reproduce

Use the development query and source projections from the linked first-query benchmarks. The source JSONL contains the original-order development rows serialized with fields id, expert_id, text and split. Use the existing pinned checkpoints and a new output path on a Slurm CUDA node.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_cross_precision_search \
  DEVELOPMENT_QUERIES DEVELOPMENT_SOURCES CHECKPOINT_DIRECTORY NEW_RESULTS
```
