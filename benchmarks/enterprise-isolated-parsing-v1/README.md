# Sharing complete request context across clause decisions

Use the frozen Qwen3-4B-Instruct-2507 on the existing 72 bilingual heading controls and the same 80 development requests as the earlier [scope study](../enterprise-heading-roles-v1/README.md). Every arm scores every clause. The public controls label one target clause per request. Enterprise role judgments are unavailable.

The three full-context arms use the same token sequences for the unmodified request and each source-anchored clause. Independent batching repeats the request per clause. The prefix-cache control computes the request state once and reuses it across clause batches. The packed arm represents the request once per block of up to eight clauses. Each suffix attends to the request and its own preceding tokens, with position indices reset to the corresponding standalone sequence. The local control uses the target and adjacent clauses with the earlier prompt layout.

A small two-layer FP32 Qwen3 check on CUDA matched the independent logits within 0.000001 for both shared implementations. Changing one packed suffix left the other three logits unchanged. The production-sized BF16 runs separately report actual label differences. Logit and timing differences include the model's numerical dependence on input shape and attention execution.

## RTX 5090 results

| Clause execution | Development P95 | Development clauses agreeing with independent batch / 1,452 | Qualification-status changes / 1,452 | Correct control targets / 72 |
| --- | ---: | ---: | ---: | ---: |
| Full context, independent batched | 5,269 ms | 1,452 | 0 | 60 |
| Full context, prefix cache | 442 ms | 1,436 | 3 | 60 |
| Full context, isolated packing | 852 ms | 1,433 | 4 | 60 |
| Adjacent local context | 756 ms | 1,153 | 55 | 45 |

Prefix caching reduced the full-context request P95 by 11.9x. It also beat the adjacent-only control in latency on these requests. Both sharing methods preserved accuracy on the public target labels and correctly resolved both views of 24 out of 36 contrast pairs. The independent batched predictor provides agreement measurements on enterprise spans. Independent human role labels are still needed to determine which disagreements matter for expert eligibility. The cache is the stronger execution baseline here. Isolated packing reprocesses the prefix per block and pays for a dense mask.

Timing includes tokenization, mask or cache construction, inference and score extraction on one warm GPU, with rotated arm order. It excludes search, queueing, and downstream citation verification. Forward token slots describe model inputs and do not measure GPU arithmetic or memory traffic. [Measured aggregates](results.json)

## Reproduce

Export the 80 development records from the pinned enterprise corpus as JSONL with `id`, `text` and `split`, keeping their stored order. Run on a Slurm GPU compute node:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_isolated_parsing \
  DEVELOPMENT_JSONL benchmarks/enterprise-heading-roles-v1/controls.json \
  QWEN_CHECKPOINT NEW_RESULTS
```

Use `--controls-only` for the public control run with the same pinned development file. The program exports aggregates. It leaves original enterprise requests outside the source package.

Prior work establishes shared prompt attention and tree-structured attention. This experiment tests clause-role extraction under those execution choices, with exact-source anchors and task-specific numerical and quality checks. [Prompt Cache, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/a66caa1703fe34705a4368c3014c1966-Abstract-Conference.html) · [DeFT, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a6df53f082619d02b9fad64a022e5de3-Abstract-Conference.html)
