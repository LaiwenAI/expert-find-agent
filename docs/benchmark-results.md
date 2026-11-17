# Benchmark Results

Historical research entry point; the current task definition, evidence boundaries, and latest results are governed by the [SIGIR English LaTeX main manuscript](paper/sigir/main.tex). Insufficient data is not treated as a negative label for actual capability.
Task: recover known associated experts from held-out paper titles. Proxy labels come from paper attribution, not human judgments of relevance.
Corpus SHA256: `925c6e60d3db3d6f7d13ff3f8bd850d2e0202b2205c0281314c6625773a50537`; random seed 42.

## All baseline queries

| Method | Number of queries | nDCG@10 | MRR@10 | Recall@10 | Average seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25-max | 30 | 0.3848 | 0.3541 | 0.5333 | 1.482 |
| bm25-vote | 30 | 0.4052 | 0.3572 | 0.6333 | 1.439 |
| tfidf-max | 30 | 0.3274 | 0.3233 | 0.3833 | 1.452 |

## Evaluation query subset shared with the same model

| Method | Number of queries | nDCG@10 | MRR@10 | Recall@10 | Average seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25-max | 10 | 0.5642 | 0.5147 | 0.8500 | 1.508 |
| bm25-vote | 10 | 0.5675 | 0.5083 | 0.8500 | 1.436 |
| hybrid-evidence-tree | 10 | 0.5551 | 0.5500 | 0.6000 | 16.451 |
| tfidf-max | 10 | 0.5244 | 0.5500 | 0.5500 | 1.456 |

Failed model experiments: 0. Failures are counted as zero scores; successful requests alone must not be aggregated.

Paired bootstrap (2,000 iterations, seed=42): tree agent minus BM25-vote difference in nDCG@10.
Mean difference -0.0124, 95% bootstrap interval [-0.2631, 0.2339]. Results from this small sample should not be extrapolated.

Total prompt tokens for successful requests: 312027; completion tokens: 6754. No price list was configured, so monetary cost is not estimated.
Recall of known authors in the candidate set for successful requests: 0.9000; this constrains the attainable ceiling for subsequent ranking.

## Interpretation limitations

Author attribution does not include all relevant experts; Recall is the recovery rate for known labels. Evidence corresponding to the title and all fields containing that complete title were excluded; related papers and paraphrased content may still provide information. Aggregate set statistics still include excluded documents.
Baselines ran sequentially in the same process, and runtime includes caching effects; model runtime is also affected by service load. Without human gold labels and evidence of statistical significance, these results cannot support claims of superiority on actual enterprise needs.
The tree agent's first-stage candidates are limited by BM25. Subsequent model verification may improve explanations, but may also cause ranking losses; per-query results should be examined.
Raw data: artifacts/benchmark/runs.jsonl; model-stage outputs and sources: artifacts/agent-benchmark/paper-*.json.

## Conclusion for this run

On the same 10-query subset, the tree agent's nDCG@10 was 0.5551, below BM25-vote's 0.5675; the paired interval includes zero, so there is no evidence of superiority. The model path is slower and consumes additional tokens. Its verified functions are checking evidence along an expert evidence tree, providing original-text sources, and listing missing requirements. The current recommendation is to use BM25-vote as a fast retrieval entry point and, when needed, use the agent to provide verification explanations. This recommendation is based only on engineering trade-offs in the current proxy experiment, and does not prove recommendation effectiveness for actual needs.
