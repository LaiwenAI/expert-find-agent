# Route uncertain reviewer searches to a centered representation

Use the 122 KDD and 59 SciRepEval queries from the [fixed profile pooling comparison](../cof-global-profile-pooling-v1/README.md). The frozen embedding model encodes each query once per arm. Compute a background vector from indexed source vectors only, subtract it from the source vectors and query, and L2-normalize the residuals. This creates another view of the same corpus and uses neither reviewer labels nor a language-model call to expand the query. It tests simple centering in an already trained embedding space. [Related work on post-hoc embedding alignment](https://aclanthology.org/2026.emnlp-main.2323/)

Compare the raw best source, raw centroid–best-source blend, centered centroid–best-source blend, fixed reciprocal-rank fusion of the two blends, and selective fusion. The selective rule takes the raw blend if at least two of its first three experts also appear in the raw best-source first three. Otherwise it computes the centered view and fuses. No thresholds were fitted to these data. All methods return ten reviewers and attach citations to the first five, with at most three candidate sources per reviewer. Centered and fused routes order citations with centered cosine or summed raw and centered cosine, respectively. A frozen cross-encoder checks sources, and citations require two model-positive judgments.

Re-run all arms on one GPU with arm order rotated across requests. Warm processing covers live embedding, full-inventory retrieval, tokenization, model checks and ranking. Costs exclude serving transport, initialization and index construction. The extra normalized source matrix is resident in memory even when a query skips its use. Known-positive recall and precision use the released partial reviewer judgments. A selected citation does not establish independently judged support.

## RTX 5090 results

| Panel | Ranker | Known-positive Recall@10 | Known-positive P@5 | Positive and cited P@5 | Retrieval P95 | Complete P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KDD | Best source | 13.64% | 3.77% | 2.13% | 9.49 ms | 126.24 ms |
| KDD | Raw blend | 18.60% | 3.93% | 1.97% | 8.99 ms | 134.75 ms |
| KDD | Centered blend | 18.60% | 3.93% | 1.31% | 9.44 ms | 132.55 ms |
| KDD | Fixed fusion | 19.83% | 3.44% | 1.48% | 9.65 ms | 134.83 ms |
| KDD | Selective fusion | 19.01% | 3.61% | 1.80% | 9.31 ms | 134.39 ms |
| SciRepEval | Best source | 50.98% | 16.61% | 8.47% | 8.87 ms | 121.62 ms |
| SciRepEval | Raw blend | 53.59% | 17.63% | 7.12% | 8.55 ms | 121.32 ms |
| SciRepEval | Centered blend | 45.10% | 15.59% | 7.12% | 9.19 ms | 131.31 ms |
| SciRepEval | Fixed fusion | 48.37% | 16.61% | 6.78% | 9.16 ms | 126.60 ms |
| SciRepEval | Selective fusion | 50.98% | 17.29% | 6.78% | 8.97 ms | 124.22 ms |

Job 41853 ran both panels. Selective fusion scored the centered view for 62/122 KDD requests and 28/59 SciRepEval requests. It lowered retrieval P95 relative to fixed fusion on both panels and kept it below 10 ms. KDD recovered one additional known positive at ten compared with the raw blend, with fewer known positives at five and fewer cited positives. On SciRepEval, selective fusion recovered four fewer known positives at ten than the raw blend and also delivered fewer cited positives. Known-positive P@5 difference intervals against best source include zero for all alternative arms in both panels. This result leaves raw retrieval as the stronger control for the proposed adaptive branch. The released judgments, reused cohorts and absent citation annotations limit broader quality claims. [KDD aggregates](kdd.json), [SciRepEval aggregates](scirepeval.json)

Run with the pinned snapshots and open checkpoints in the profile pooling study on a Slurm CUDA node.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.cof_contrastive_shortlist KDD EMBEDDING RERANKER NEW_KDD_RESULTS
python -m experiments.retrieval.cof_contrastive_shortlist SciRepEval EMBEDDING RERANKER NEW_SCIREPEVAL_RESULTS
```
