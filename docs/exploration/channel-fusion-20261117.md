# Source-channel fusion diagnostic

On 1,000 paired construction queries, we compared the original global expert ranking with separate paper and profile retrieval followed by reciprocal rank fusion (RRF) or interleaving. These are standard controls, not new algorithms. Each channel contributes at most its top 50 experts before fusion. The same cached embeddings and expert corpus were used throughout.

| Intent type | Pairs | Global overlap@10 | Exact RRF overlap@10 | Exact interleave overlap@10 |
|---|---:|---:|---:|---:|
| Overview | 200 | 0.3800 | 0.4710 | 0.4495 |
| Component method | 200 | 0.4260 | 0.4990 | 0.4680 |
| Conditional transfer | 200 | 0.4590 | 0.5280 | 0.4880 |
| Evidence lookup | 200 | 0.4135 | 0.4740 | 0.4450 |
| Constraint focused | 200 | 0.4295 | 0.5085 | 0.4715 |

RRF increases language overlap in each category, but also strongly favors experts represented in both channel shortlists: 16,668 of its 20,000 returned slots have both channel ranks, compared with 6,727 for interleaving. This describes shortlist membership, not a causal estimate of bias or evidence quality. Missing a channel is not evidence that an expert lacks the ability to solve a request.

Approximate channel retrieval with RRF has overlap 0.4962 and retains 99.545% of the exact fused top ten on average. Its mean retrieval-plus-fusion time is 13.52 ms per language view; this excludes query encoding, index construction and loading. These values do not establish end-to-end latency or expert relevance. Translation quality and independent expert-fit labels remain unresolved, so neither overlap nor shortlist membership should be reported as accuracy.

The local artifact `artifacts/bilingual-fusion-diagnostic-v1/` contains the frozen protocol, per-pair rankings and summary with hashes. The diagnostic source is `experiments/bilingual/fusion_diagnostic.py`.
