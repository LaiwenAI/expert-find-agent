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

## Dynamic-shape controls

Job `40593` repeats the fixed-shape references alongside dynamic batches that pad only to their current longest row. The stricter fill strategy takes consecutive extra sources only while their token length fits that width. All strategies prepare the same initial candidate prompts. Token slots include real and padding tokens.

| Schedule | One-source P95 | Two-source P95 | Mean token slots with two sources | Exact ordered results vs fixed exhaustive, one / two |
| --- | ---: | ---: | ---: | ---: |
| Fixed shape, one per expert | 263.20 ms | 239.85 ms | 15,676 | 73 / 73 |
| Fixed shape, fill idle slots | 235.53 ms | 226.53 ms | 14,785 | 73 / 73 |
| Dynamic shape, one per expert | 200.08 ms | 179.95 ms | 10,871 | 71 / 71 |
| Dynamic shape, fill available slots | 198.79 ms | 185.69 ms | 11,602 | 71 / 71 |
| Dynamic shape, fill within current width | 198.10 ms | 185.32 ms | 11,348 | 71 / 71 |

Each column compares the same 73 queries. All three dynamic policies agree with one another on every ordered result. Each differs from the fixed-shape reference on two queries per support setting. With two required sources, filling slots uses fewer calls and processes more token slots, increasing P95 by 5.75 ms relative to dynamic one-per-expert. The paired descriptive interval for the P95 reduction is −19.52 to +5.46 ms. Limiting the sequence width alone leaves the cost of adding rows. The faster dynamic baseline warrants its own comparison when the application permits a change in decision semantics. [Raw aggregates and paired intervals](dynamic-shape-v1.json)

## Selective fixed-shape rechecks

Job `40600` scores each wave using dynamic shapes, then repeats only pairs with an absolute logit margin at most 0.25 in the original four-slot shape. The fixed-width reference scores replace those pairs before the next expert scheduling decision. The same band is used on reused SIGIR queries and on SciRepEval queries after excluding profile overlaps. The two query sets are disjoint by paper ID and normalized title. In each SciRepEval setting, eight prepared candidate prompts across five queries hit the shared token cap. Timing includes each extra GPU call.

| Dataset | Required sources | Dynamic exact | Guarded exact | Fastest fixed-shape P95 | Guarded P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIGIR | 1 | 71/73 | 73/73 | 235.54 ms | 209.12 ms |
| SIGIR | 2 | 71/73 | 73/73 | 226.49 ms | 207.17 ms |
| SciRepEval | 1 | 85/91 | 91/91 | 265.74 ms | 242.24 ms |
| SciRepEval | 2 | 87/91 | 91/91 | 316.40 ms | 261.60 ms |

Exactness includes expert order and cited source IDs. The guard reproduces all reference outputs and lowers P95 by 8.5% to 17.3% relative to the fastest fixed-shape arm. It is slower than unguarded dynamic scoring in all four comparisons. SciRepEval with one required source has one residual decision-sign difference that leaves the ranking and citations intact. The finite guard is an empirical control tied to the measured checkpoint and environment. [Results](margin-guard-v1.json)

## Reproduce

Run [the GPU comparison](../../experiments/retrieval/cof_live_certified_batching.py) on a Slurm CUDA compute node with `COF_INPUT_MANIFEST EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON`. Add `--prefix-control` with a separate output path for the consecutive-source controls. The replay uses the published CoF inputs described in [the bounded adjustment benchmark](../cof-bounded-adjustment/README.md).

Add `--compare-dynamic` with a separate output path to reproduce the dynamic-shape controls.

Add `--guard-shape` to replay the selective rechecks. Add `--dataset SciRepEval` for the filtered SciRepEval cache.
