# Share the exact prompt prefix within one role-verification request

The [selective recheck benchmark](../role-gap-selective-recheck-v1)
showed that a second neural forward increased tail latency and that
preserving padded width did not guarantee identical scores. Here all
four selected sources remain fixed. Test whether sharing repeated
instruction and query tokens speeds up reranking from the first
request, including all cache or mask construction.

| Execution | Model work |
| --- | --- |
| Padded reference | Score four separate sequences in one ordinary padded forward. |
| Independent packed | Score all tokens in one ragged forward with per-sequence attention offsets. |
| Prefix KV fork | Prefill the exact shared tokens, broadcast read-only cache views, then score all suffixes together. |
| Prefix tree | Process each common token once, with one dense tree mask and a single forward. |

Find the prefix by comparing the token IDs of the original complete
reranker prompts. Leave at least one token in each suffix. Each tree
suffix attends to the common prefix and its own causal history, with
position IDs matching its standalone prompt. The KV fork uses shared
read-only prefix views and independent append operations. Check both
implementations by changing one branch at fixed shape and requiring
identical scores from the other branches. No request history or
cross-request cache is used.

Use the same Qwen checkpoints, topic ranking, role-compatible title
refill, visibility masks and cutoff as the previous PLOS and eLife
replay. Reproduce the padded citation aggregates. Report the number
of verifier launches, model input token positions, raw score
differences, cutoff crossings, exact ordered expert–source response
matches, and expert-set and witness-set changes. A token position is one model input position;
it does not represent equal attention work across these executors.
Warm request latency covers query encoding, retrieval, tokenization,
mask or cache preparation, execution and citation serialization.
It excludes cold start, network, queues and human adjudication.

## Results

Slurm jobs `44048` and `44049` agree on every non-timing metric.
The table uses the full declaration masks and the warm first-run
P95. Exact output counts compare serialized ordered witnesses with
the padded reference.

| Execution | PLOS exact / 46 | PLOS P95 | eLife exact / 110 | eLife P95 |
| --- | ---: | ---: | ---: | ---: |
| Padded | 46 | 35.64 ms | 110 | 29.54 ms |
| Independent packed | 43 | 25.96 ms | 108 | 22.74 ms |
| Prefix KV fork | 44 | 41.75 ms | 105 | 34.71 ms |
| Prefix tree | 44 | 34.29 ms | 107 | 25.04 ms |

Across both visibility masks, prefix-tree execution matches
304 of 312 responses, as does independent packing. The KV fork
matches 299. A third replay, job `44053`, reproduces those counts
and records the nature of changes. In the tree arm, six requests
change the expert set and two change witness order alone. eLife
loses one broad-topic citation at full visibility and has no net
change at partial visibility. All citations use the same
declaration binding; publisher labels are article-level proxies.

The common prefix averages approximately 54 tokens
per request. The tree removes about 12% of the ragged input
positions, while its observed P95 is consistently higher than
independent packing. The dense causal mask and different
attention layout contribute work that token counts do not capture.
The fork adds another verifier forward and is slower than the
padded route in both catalogs.

[Aggregate results](results.json)

## Reproduce

Use the frozen inputs from the [gap-refill benchmark](../role-gap-title-refill-v1).
Run on a Slurm CUDA node with PyTorch variable-length attention.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.role_gap_prefix_reuse PLOS_PROJECTION ELIFE_PROJECTION \
  benchmarks/elife-mesh-v1/mesh.json benchmarks/role-gap-title-refill-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
