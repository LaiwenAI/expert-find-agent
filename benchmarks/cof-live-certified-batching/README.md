# Certified evidence batching with a live verifier

The CoF SIGIR replay retrieves the top ten experts from the resident FP32 index. The first three sources per expert are ordered by dense score. A frozen Qwen3-Reranker-0.6B checks each source publication for methodological relevance to the query paper. The frozen ownership mapping ties each source to its expert. Experts with the required number of accepted sources rank before the remaining experts, in their original order within each group.

Each GPU call has four slots. Every schedule uses the same per-query token width, rounded to 64 tokens, and fills vacant slots with copies whose scores are discarded. This isolates schedule choices from score differences caused by changing tensor shapes. The comparison includes one check per active expert, a consecutive-source control that can displace other experts, and an idle-slot policy. The idle-slot policy schedules one check per active expert first and uses otherwise padded slots for later sources of those same experts. All used comparisons keep the canonical evidence order and apply the [exact stopping boundary](../trec-expert-certified-stop/README.md) between calls.

The timer includes query tokenization and encoding, resident retrieval, preparing candidate prompts, scoring, and result assembly. The comparison uses warm single requests. Reviewer labels assess expert relevance. They do not independently certify the relevance or personal attribution of a selected document.

## Results on RTX 5090

Job `40581` measures the consecutive-source controls. With one required source, giving each selected expert up to two positions increases mean checks from 19.96 to 22.44. Its request P95 falls from 263.24 to 243.15 ms. With two required sources, giving each expert up to three positions instead raises P95 from 239.84 to 253.57 ms.

Job `40584` measures filling idle slots on the same 73 requests. Each row gives paired measurements for one check per active expert versus idle-slot filling.

| Required accepted sources | Mean GPU calls | Mean checks | Request P95 |
| --- | ---: | ---: | ---: |
| One | 5.88 to 5.25 | 19.96 to 20.03 | 263.41 to 236.26 ms |
| Two | 6.71 to 6.33 | 22.79 to 24.21 | 240.20 to 226.47 ms |

Idle-slot filling reduces P95 by 10.3% and 5.7% in the two settings. It uses fewer GPU calls on 35 and 27 requests, respectively, with no requests using more calls. Both settings reproduce the exhaustive ordered experts and citations on all 73 requests. Every evaluated candidate–source margin agrees with the fixed-shape exhaustive reference. That equality applies to this checkpoint, hardware, and workload. The comparison uses a fixed-shape baseline and measures one warm request at a time. [Results and paired latency intervals](results.json)

## Reproduce

Run [the GPU comparison](../../experiments/retrieval/cof_live_certified_batching.py) on a Slurm CUDA compute node with `COF_INPUT_MANIFEST EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON`. Add `--prefix-control` with a separate output path for the consecutive-source controls. The replay uses the published CoF inputs described in [the bounded adjustment benchmark](../cof-bounded-adjustment/README.md).
