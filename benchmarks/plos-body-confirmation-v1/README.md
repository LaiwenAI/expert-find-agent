# Recover source coverage with bounded body verification

Start from the [45-query integrated attribution test](../plos-open-legacy-attribution-v1/README.md). Reuse the frozen 113-source inventory, exact top-ranked source, result and keyword clues, dense confidence thresholds, and author gate. Use the publisher XML body to compile windows of up to three sentences or 220 tokens according to the benchmark tokenizer. Remove sentences that share at least 70% of the shorter four-word-shingle set with either held-out abstract sentence of the same paper. This deliberately prevents short verbatim or near-verbatim evidence reuse. Figure or table content is not separately represented. Keep the abstract suffix outside the retrieval index.

Keep every source accepted by the frozen dense cutoff. When that cutoff abstains, compare three potential rescue signals for the same single candidate. The lexical route uses maximum weighted query-token coverage of a body window. The prefix reader uses the frozen Qwen3-Reranker-0.6B over title and the indexed abstract prefix. The body reader scores up to two windows selected by the lexical route and keeps the higher score. For each clue type and method, set the rescue cutoff to the maximum score among incorrect calibration proposals that the dense gate rejected. No method can add a wrong calibration source. Thresholds are carried to the held-out source queries without test tuning.

Report rescued sources and downstream authors independently. A source can be emitted without any recommended author. Use the same author outputs whenever a route admits a source. The factored latency combines measured query encoding and retrieval, the tested gate's window selection and model time when used, and author verification when accepted. It excludes index construction, request parsing, transport and concurrent queueing. The text is sourced from real publisher documents. The exact-study query and its missing-source control are constructed from held-out abstract material.

## RTX 5090 results

Slurm job 42656 evaluated 45 held-out source queries across 44 author-name groups. The catalog contains 10,005 body windows across 113 sources. Each row contains 45 source-present and 45 paired source-withheld queries. Wrong-author queries include wrong-source recommendations and can overlap with supported queries.

| Clue | Source gate | Correct source, present | Supported author, present | Wrong author, present | Wrong-source recommendation, withheld | Factored P95, present / withheld |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Excerpt | Dense | 34 | 31 | 9 | 0 | 82.20 / 7.57 ms |
| Excerpt | Dense + lexical body | 42 | 39 | 11 | 0 | 97.93 / 7.81 ms |
| Excerpt | Dense + prefix reader | 44 | 41 | 11 | 2 | 97.93 / 16.05 ms |
| Excerpt | Dense + body reader | 44 | 41 | 11 | 1 | 97.93 / 19.44 ms |
| Keywords | Dense | 22 | 20 | 9 | 2 | 80.05 / 7.26 ms |
| Keywords | Dense + lexical body | 27 | 25 | 9 | 2 | 80.05 / 7.33 ms |
| Keywords | Dense + prefix reader | 33 | 31 | 11 | 3 | 81.40 / 30.50 ms |
| Keywords | Dense + body reader | 28 | 26 | 10 | 3 | 80.99 / 41.48 ms |

Lexical body coverage recovered 8 excerpt queries and 5 keyword queries with a supported author, with the same observed number of recommendations from wrong sources on the withheld queries. The descriptive paired 95% intervals for the support gains were +6.82 to +29.55 and +2.27 to +20.45 percentage points, respectively. The bootstrap uses author-name clusters despite the generic sampling-unit label in the aggregate. For excerpts the lexical route also produced wrong authors on two additional queries with the source available, and raised the number of 4B author calls from 41 to 54. For keywords the prefix reader recovered more support than reading the body, and both readers introduced an additional wrong-source recommendation on the withheld queries.

These results favor testing the lexical check as a low-cost fallback for source identity on a fresh split. The source catalog, authors and query formats reuse the earlier test panel. Equal observed wrong-source counts do not establish equal risk. The published check still ranks by title and abstract, confirms only its first candidate, and targets a specific study. The measured effects apply to source confirmation and subsequent attribution with frozen models. [Aggregate results](results.json)

## Reproduce

Use the same captured corpus and open checkpoints as the integrated test, plus the frozen Qwen3-Reranker-0.6B. Run inside a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.legacy_body_confirmation \
    OLD_SNAPSHOT INDEPENDENT_SNAPSHOT QWEN_EMBEDDING QWEN_RERANKER QWEN4B \
    OPEN_ATTRIBUTION_RESULTS CASCADE_RESULTS NEW_RESULTS
```
