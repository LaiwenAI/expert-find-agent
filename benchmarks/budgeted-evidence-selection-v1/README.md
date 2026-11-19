# What a fixed sentence budget can recover

Compare a two-window lexical selector, within-source BM25 sentence ranking, and weighted coverage. Weighted coverage keeps the top BM25 sentence and then greedily adds sentences that cover new query terms and adjacent query word pairs, weighted by within-source inverse document frequency. All compact methods use the same number of sentences for each source. The lexical selector sets this budget at up to six. Every selected excerpt preserves source order. Gold sentence positions are used only to evaluate the selected evidence.

## SciFact-Open development comparison

Of 460 annotated pairs, exclude 23 without sentence locations and 38 whose abstracts exceed the previously fixed 2,500-character cap after that exclusion. The 399 remaining pairs have 211 support and 188 contradiction labels. The cohort includes the 37 pairs in the [earlier rationale audit](../scifact-compact-open-rationale-v1/README.md). Other pairs can share claims or documents with past experiments. This is an exploratory comparison on provided abstracts. Use the same frozen Qwen3-4B reader and unchanged thresholds, including the same compact threshold for all three selectors. [SciFact-Open, Findings of EMNLP 2022](https://aclanthology.org/2022.findings-emnlp.347/)

| Evidence view | Complete annotated sets / 399 | True supports accepted / 211 | False supports accepted / 188 | Mean evidence tokens | Verifier P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full abstract | 399 | 121 | 11 | 308.1 | 38.45 ms |
| Lexical windows | 217 | 122 | 14 | 157.9 | 27.74 ms |
| BM25 sentences | 237 | 132 | 14 | 160.8 | 27.81 ms |
| Weighted coverage | 233 | 133 | 14 | 161.2 | 27.82 ms |

BM25 gains 58 complete sets and loses 38 relative to lexical windows. The paired difference is +5.0 percentage points, with a connected-component bootstrap interval of +0.2 to +9.7 points. The net gain is 19 in the 178 citation pairs and one in the 221 pooled pairs. In the earlier 37-pair cohort, both new selectors recover 18 complete sets versus 20 for lexical windows. All compact methods accept three more contradictions than the full reference. Latency covers sentence selection, tokenization and batch-one verification from a provided claim and abstract on an RTX 5090. Evidence token counts exclude the rest of the prompt.

## NLI4CT transfer

Apply the same selection rules without tuning to the 500 official NLI4CT gold-test statements. Trial IDs and the annotated task section are provided. For comparison questions, complete evidence requires all annotated lines from both trials. This transfer measures evidence selection only. [NLI4CT, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.1041/)

| Evidence selector | Complete / 500 | Single trial / 229 | Two trials / 271 | Sentence micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Lexical windows | 80 | 48 | 32 | 53.0% |
| BM25 sentences | 67 | 43 | 24 | 51.7% |
| Weighted coverage | 66 | 42 | 24 | 51.7% |

All methods select 6.76 sentences per statement on average across the supplied trial sections. Only 147 of 500 statements fit their per-trial sentence budgets even with perfect selection. Allowing an oracle to redistribute slots across the known trial sections raises the ceiling to 157 of 500. This ceiling is defined by complete annotation coverage. It does not estimate answer correctness or the sufficiency of unlabeled alternatives.

The SciFact-Open BM25 gain does not transfer to NLI4CT. These diagnostics make BM25 a useful short-context baseline and motivate comparing adaptive evidence depth against fixed budgets. Neither study evaluates expert retrieval or end-to-end service quality.

## Reproduce

Use the [original SciFact-Open release](https://github.com/dwadden/scifact-open), the prior [37-pair projection](../scifact-compact-open-v1/README.md), and the [NLI4CT gold-test archive](https://github.com/ai-systems/nli4ct/blob/8e8e5fc506b916984011873403b9ea5ed6dd96c2/Complete_dataset.zip). Use new output paths. Only the SciFact-Open reader evaluation requires a GPU.

```sh
PYTHONPATH=src:. python -m experiments.scifact.budgeted_sentence_selection prepare \
  SCIFACT_OPEN_ARCHIVE EARLIER_37_PAIRS NEW_PAIRS
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.budgeted_sentence_selection run \
  NEW_PAIRS QWEN_4B_CHECKPOINT NEW_SCIFACT_RESULTS
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m experiments.scifact.nli4ct_selector_transfer \
  NLI4CT_ARCHIVE NEW_NLI4CT_RESULTS
```

Job 41344 ran the frozen reader on ws5. [SciFact-Open aggregates](scifact-open.json) · [NLI4CT aggregates](nli4ct.json)
