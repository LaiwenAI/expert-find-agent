# Bilingual retrieval diagnostic, 2026-11-17

This completed construction diagnostic uses one derived intent per original enterprise parent: 1,000 pairs from 1,000 connected company groups, five intent types with 200 pairs each. There are 860 pairs retained by the model/numeric construction screen and 140 unjudged pairs. These are derived search intents, not complete original English translations or relevance gold.

Both languages search the same frozen index of 96,749 owned source representations and 9,861 experts. The source encoder and source-view boundaries are inherited unchanged from the full-corpus baseline. Source/vector/checkpoint hashes are verified on the Slurm compute node; none of the 2,000 query views is truncated. A full-source representation is still bounded by its original encoder/input protocol, not the entire full text of every publication.

| Method | Mean CPU query ms | Exact top-ten entries retained | Identical ordered lists / 2,000 |
| --- | ---: | ---: | ---: |
| Exact | 33.138 | 100% | 2,000 |
| HNSW floor64 | 1.425 | 94.690% | 1,327 |
| HNSW floor512 | 3.373 | 99.805% | 1,962 |

These timings use one call per view after warmup, alternating method order, and one FAISS thread. They exclude index construction, query encoding, source loading, and LLM assessment. Graph construction and index copies take59.10s. Batched encoding of all2,000 views takes2.73s after load/warmup on an allocated RTX4090; this is throughput timing, not single-request latency.

Exact Chinese/English top-ten overlap averages42.16%, with company-group bootstrap interval[41.03%,43.30%]. The screened subset averages42.36%; unjudged pairs40.93%. Mean query-vector cosine is0.8103, and no pair shares an identical ordered top ten. By intent, exact overlap is38.00% overview,42.60% component method,45.90% conditional transfer,41.35% evidence lookup, and42.95% constraint-focused.

The high-effort ANN has42.10% language overlap, a difference of−0.06 percentage points from exact, interval[−0.13,+0.01]. A small difference or an interval crossing zero is not an equivalence test. Nevertheless, the observed approximation loss cannot account for most of the much larger exact cross-language disagreement. Translation differences, encoder effects and narrow score margins are competing explanations; this experiment does not isolate them or show which language ranks correctly. Consistent wrong rankings could have perfect overlap.

Authoritative local files are `artifacts/bilingual-retrieval-v1/{selection,protocol,summary,analysis,transfer-verification}.json`, `evaluation.jsonl`, and `query-vectors.npz`. Job39435 completed on ws6,0:0,2min51s. The local transfer checks gzip CRC, allowed paths, and protocol/evaluation/query-vector hashes. Reproduction code is under `experiments/bilingual/`, with `experiments/cluster_bilingual_retrieval.sbatch`. This is a larger fidelity and bilingual-stability study, not whole-agent efficiency or independent industrial accuracy.
