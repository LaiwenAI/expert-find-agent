# Wide-pool stopping for live reviewer verification

Use all 73 queries and the released soft and hard reviewer labels from the [CoF SIGIR split](https://github.com/yuzhimanhua/CoF#datasets). For each request, a live frozen Qwen3 embedding and a resident exact index select the top 32 reviewers by maximum source similarity from 189 reviewers and 1,740 profile publications. The benchmark checks that no query paper ID is present in the source index. Only the best-scoring owned publication is used per reviewer.

The frozen Qwen3-Reranker-0.6B checks methodological relevance of each publication. The filter accepts a nonnegative yes-minus-no margin and selects up to five passing reviewers in dense prior order. Compare an all-32 batch, four exhaustive batches of eight, the same batches of eight with prefix stopping, and stopping after one batch of eight with a batch of 24 on fallback. All arms freshly encode the same query and reproduce its ordered candidate-source pairs. An unchecked suffix cannot change the first five positive decisions once they are found, conditional on keeping batch-sensitive verifier decisions fixed.

## Results

Job `40462` ran on an RTX 5090. There were no exact normalized title or title-plus-abstract matches between query papers and profile sources. The measured request includes query encoding, full source search, model calls, and result assembly after warm-up.

| Verification policy | Mean checks | Mean batches | Mean request | P95 request | Hard P@5 | Soft P@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All 32 in one batch | 32.00 | 1.00 | 221.75 ms | 275.69 ms | 13.70% | 29.32% |
| Exhaustive batches of eight | 32.00 | 4.00 | 187.42 ms | 222.55 ms | 14.25% | 29.32% |
| Prefix stopping, batches of eight | 22.47 | 2.81 | 133.94 ms | 213.50 ms | 14.25% | 29.32% |
| Stop after eight, otherwise verify remaining 24 | 24.77 | 1.70 | 158.77 ms | 236.09 ms | 14.25% | 29.32% |

With fixed batches of eight, stopping preserves all 73 ordered reviewer-source lists and all common-prefix verifier decisions. It skips at least one batch on 35 requests, cuts mean checks by 29.8%, and reduces P95 by 4.1% relative to the fastest exhaustive verification schedule measured here. The all-32 batch changes the selected results on one request. Comparing stopping against that batch alone would conflate batch sizing with skipped checks.

Dense retrieval without a verifier returns hard/soft P@5 of 20.00%/42.19%. Sorting all 32 by the verifier margin yields 20.00%/44.38%. The acceptance filter returns about 3.40 reviewers per request. Empty slots count as misses. Stopping preserves the filtered policy and incurs a quality deficit relative to the unfiltered baseline on the released reviewer labels. These labels assess reviewer suitability and provide no independent judgment of whether a specific cited source justifies the match.

## Reproduce

Build the official frozen source vectors with [the CoF embedding runner](../../experiments/cof_qwen_embedding.py). Run the lexical baseline (`python -m expert_search.cof_benchmark`) to prepare the companion manifest inputs. Build the [GPU input manifest](../../experiments/retrieval/prepare_cof_gpu_inputs.py) as in the previous [GPU quality diagnostic](../../experiments/retrieval/cof_gpu_quality_diagnostic.py). The live runner checks the CoF source and label files and the cached source vectors against that manifest.

Run [the live benchmark](../../experiments/retrieval/cof_live_prefix_stop.py) with `INPUT_MANIFEST EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON` in the repository root on a Slurm CUDA compute node.
