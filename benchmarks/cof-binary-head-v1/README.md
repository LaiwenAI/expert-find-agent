# Two-token output projection for CoF evidence verification

The verifier uses a causal Qwen3-Reranker-0.6B decoder and consumes only the final-token yes and no logits. The baseline still projects the final state over the full vocabulary. Earlier local MLX code used a compact projection. This experiment checks the same optimization in the CUDA shared-frontier verifier on two public expert-search panels.

Generate the top-five experts and up to three source records per expert once per request, using the same full-inventory profile-and-best-source blend as the [profile-first benchmark](../cof-profile-first-v1). All verifier arms receive those identical lists. The shared frontier uses two supportive sources, batch width four, and the existing 0.25-logit guard. Fixed-shape boundary rechecks always use the original full projection.

| Arm | Dynamic forward | Prompt preparation |
| --- | --- | --- |
| Full | Full-vocabulary head, then select the yes and no logits | One tokenizer call per prompt |
| Two rows | BF16 linear projection using just the yes and no weight rows | One tokenizer call per prompt |
| Two rows + batched tokenization | Same compact projection | One tokenizer call for all pending prompts |

The three arms use the same frozen backbone, prompt format, guarded decisions and scoring schedule. Mathematical equivalence of the projection in real arithmetic does not ensure equality across different GPU kernels. The replay checks prompt token maps, compared margins, selected experts, and ordered citations, and retains any discrepancy in the aggregate. It exercises the original full projection for rechecks.

Warm measured P95 covers prompt preparation, model verification and the frontier scheduler. It excludes query encoding, candidate retrieval, model load, request handling and queueing. Arm order rotates across requests. The benchmark pairs each route with the full head on the same candidate lists. Exact output agreement is a numerical and execution check against this verifier; it gives no new evidence about whether the citations are valid or whether they belong to a contributor.

## Results

Slurm job 42776 evaluated 122 KDD and 59 SciRepEval requests on one RTX 5090. The compact head uses two of 151,669 output weight rows.

| Arm | KDD verifier P95 | SciRepEval verifier P95 | KDD exact outputs | SciRepEval exact outputs |
| --- | ---: | ---: | ---: | ---: |
| Full | 126.57 ms | 113.69 ms | 122/122 | 59/59 |
| Two rows | 125.56 ms | 112.29 ms | 122/122 | 59/59 |
| Two rows + batched tokenization | 124.82 ms | 111.87 ms | 122/122 | 59/59 |

Prompt token maps match on all 181 requests. Margin maps match on 180; the maximum common-pair difference is 0.125 logits on SciRepEval. Boundary rechecks, forward counts and padded token counts match. The two-row projection alone reduces mean verifier latency by 0.76 ms on KDD (paired 95% interval 0.63 to 0.92 ms saved) and 0.94 ms on SciRepEval (0.74 to 1.22 ms saved). Batched tokenization saves a small additional amount. This supports the compact projection as an execution optimization. A larger efficiency gain requires changing backbone effort or scheduling while retaining response quality.

[KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json)

## Reproduce

Use the same pinned CoF KDD and SciRepEval inputs and model directories as the [full-inventory profile benchmark](../cof-global-profile-pooling-v1). Run on a Slurm CUDA node.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_binary_head \
  KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_KDD_RESULTS
PYTHONPATH=src:. python -m experiments.retrieval.cof_binary_head \
  SciRepEval EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_SCIREPEVAL_RESULTS
```
