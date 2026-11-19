# Live verification after adaptive evidence selection

Keep the [offline-trained NLI4CT controller](../nli4ct-adaptive-depth-v1/README.md), its stopping cutoff and the 500 gold-test statements fixed. Compare the initial lexical windows, two or three forced learned expansions, the learned stop policy, and all lines in the supplied trial sections. Every arm uses the same pinned Qwen3-4B-Instruct-2507 checkpoint and a single judgment per statement. The reader receives the statement, section name and source-ordered excerpts, with primary and secondary trials labeled separately. [NLI4CT, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.1041/)

The prompt offers entailment, contradiction and insufficient evidence. The primary binary accuracy compares the entailment and contradiction logits. Unconstrained choice among the three logits is reported as a separate abstention diagnostic. The benchmark labels the original sources with 250 entailments and 250 contradictions. A short excerpt may justify abstention even if the full sources support a binary judgment.

| Route | Complete evidence | Binary judgments correct | Macro-F1 | Mean input tokens | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial windows | 80/500 | 301/500 | 56.4% | 291.0 | 35.26 ms |
| Fixed two expansions | 264/500 | 322/500 | 61.6% | 376.9 | 41.09 ms |
| Fixed three expansions | 317/500 | 324/500 | 62.5% | 410.1 | 44.49 ms |
| Adaptive | 268/500 | 311/500 | 59.2% | 348.0 | 37.57 ms |
| All supplied section lines | 500/500 | 333/500 | 64.4% | 582.3 | 77.57 ms |

Against forced two-step expansion, the adaptive route gains seven correct judgments and loses 18. The net difference is −2.2 percentage points, with a paired 95% interval of −4.7 to −0.2 across 113 connected components of statements and trial IDs. Its observed P95 falls 8.6%, with a fixed-to-adaptive P95 ratio of 1.09× and a 95% interval of 1.00–1.15× (unrounded lower endpoint 0.995). Against the complete-section arm, P95 falls 51.6% and binary accuracy falls 4.4 points. When the insufficient-evidence choice is available, the adaptive reader abstains 120 times, compared with 98 for two fixed expansions and 66 with the complete section.

The evidence-coverage advantage over two fixed expansions does not translate into judgment accuracy. This separates an evidence-location objective from a verifier-aware stopping objective. Timings include live selection, prompt construction, tokenization and CUDA inference at batch size one, starting from supplied source sections. Cases rotate the arm order. Source retrieval, expert ranking and service latency remain separate evaluation tasks.

## Reproduce

Use the [pinned official archive](https://github.com/ai-systems/nli4ct/blob/8e8e5fc506b916984011873403b9ea5ed6dd96c2/Complete_dataset.zip), controller result JSON from the linked study, and [pinned reader checkpoint](../scifact-live-selective-v1/README.md). Run within a CUDA Slurm allocation with a new output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.nli4ct_adaptive_reader \
  NLI4CT_ARCHIVE FROZEN_CONTROLLER QWEN_4B_CHECKPOINT NEW_RESULTS
```

Job 41374 ran on an RTX 5090 on ws5. [Aggregate results](results.json)
