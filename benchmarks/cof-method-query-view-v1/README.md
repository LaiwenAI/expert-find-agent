# Separate retrieval query framing from original-query evidence checks

Reuse the 122 KDD and 59 SciRepEval queries from the [fixed profile pooling study](../cof-global-profile-pooling-v1/README.md). Use the same complete reviewer inventories, source vectors, open embedding model, and open reranker. All routes retrieve ten reviewers and attach citations to the first five using at most three source candidates each. A citation result requires two model-positive sources. Every verifier receives the complete original query, even when the retriever uses an excerpt.

The raw best-source and raw centroid–best-source blend are the controls. The three proposed blend arms share an instruction emphasizing prior methods and evaluation expertise. They encode the whole query, the title and first three abstract sentences, or the title and three sentences with the most distinct method cue matches, respectively. Matches cover terms such as algorithm, training, implementation and experimentation. The selected sentences retain original order. The lead and method arms use the same sentence limit and can have different character and token lengths. The three arms use the same single encoder call and no query-generation model.

Rotate the order for sequential warm measurements. Report live query encoding, retrieval, tokenization, verification and ranking. Index construction and serving transport are outside the timer. Public judgments cover only part of the returned expert slots. Known-positive metrics use them as released, and citation presence does not establish independently verified citation support.

## RTX 5090 results

| Panel | Search view | Known-positive Recall@10 | Known-positive P@5 | Positive and cited P@5 | Retrieval P95 | Complete P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KDD | Raw best source | 13.64% | 3.77% | 2.13% | 9.07 ms | 126.60 ms |
| KDD | Raw blend | 18.60% | 3.93% | 1.97% | 8.64 ms | 134.30 ms |
| KDD | Method instruction, full | 18.60% | 3.44% | 1.48% | 8.63 ms | 136.52 ms |
| KDD | Method instruction, lead excerpt | 14.05% | 2.95% | 1.64% | 8.48 ms | 128.85 ms |
| KDD | Method instruction, method excerpt | 14.46% | 3.11% | 1.15% | 8.63 ms | 137.98 ms |
| SciRepEval | Raw best source | 50.98% | 16.61% | 8.47% | 8.73 ms | 121.75 ms |
| SciRepEval | Raw blend | 53.59% | 17.63% | 7.12% | 8.44 ms | 120.61 ms |
| SciRepEval | Method instruction, full | 54.25% | 17.63% | 6.44% | 8.43 ms | 124.16 ms |
| SciRepEval | Method instruction, lead excerpt | 51.63% | 16.95% | 7.46% | 8.35 ms | 126.83 ms |
| SciRepEval | Method instruction, method excerpt | 45.75% | 15.25% | 6.78% | 8.41 ms | 127.79 ms |

Job 41926 uses case-independent segmentation. An earlier execution used a splitter that expected a capital at the start of each sentence, which left the lowercase excerpts effectively unshortened. The corrected method excerpts average 579 versus 1,434 full-query characters on KDD, and 530 versus 1,004 on SciRepEval. Both raw controls exactly reproduce their prior known-positive and citation metrics.

The full-query method instruction retrieves one more known positive at ten on SciRepEval and no net increase on KDD relative to the raw blend. Positive-and-cited precision falls in both. Cue-based excerpts lower recall on both panels and increase complete P95 relative to the raw blend. The same-depth lead excerpt has lower complete P95 and more cited positives than the method excerpt on both panels. These results establish no gain from replacing the full query with a cue-selected excerpt. [KDD aggregates](kdd.json), [SciRepEval aggregates](scirepeval.json)

Run with the same pinned inputs and checkpoints as the profile pooling study on a Slurm CUDA node.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.cof_method_query_view KDD EMBEDDING RERANKER NEW_KDD_RESULTS
python -m experiments.retrieval.cof_method_query_view SciRepEval EMBEDDING RERANKER NEW_SCIREPEVAL_RESULTS
```
