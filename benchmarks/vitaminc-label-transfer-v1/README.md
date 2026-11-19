# Test verifier label-order transfer on VitaminC

Freeze the combined digit and display-order swap observed on NLI4CT before applying it to the [VitaminC fact verification test file](https://huggingface.co/datasets/tals/vitaminc). Use the author's pinned release listed in the aggregate. Select 150 support–refute pairs and 150 support–insufficient-evidence pairs from real revisions. Each pair shares the source case and verbatim claim, with distinct excerpts. Deterministic hash selection keeps one pair per Wikipedia page and never repeats a case-insensitive claim. The resulting 600 items are a paired diagnostic with twice as many support examples as either other class. [Dataset paper](https://aclanthology.org/2021.naacl-main.52/)

The frozen Qwen3-4B reader sees only the query and supplied evidence. Compare the four fixed prompt variants from the [NLI4CT experiment](../nli4ct-label-bias-v1/README.md). Replace clinical-specific wording with a fixed Wikipedia preamble for every arm, while preserving each semantic mapping and label order. Keep the insufficient-evidence digit fixed. Score the three logits after mapping them back to the semantic labels. No parameters, thresholds or variants are selected using this panel. Rotate arm order and report single-call P95 over tokenization and inference.

Measure exact pair accuracy as well as item accuracy, macro F1, false support on explicit refutations, and false support on insufficient evidence. Confidence intervals resample whole pairs. This is transfer across known-evidence corpora with the same model. It does not establish generalization across readers or to expert attribution.

## RTX 5090 results

| Prompt | Correct items / 600 | Both right / 300 | Support–refute / 150 | Support–insufficient / 150 | Refute → support / 150 | Insufficient → support / 150 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 382 | 106 | 63 | 43 | 18 | 22 | 14.81 ms |
| Swap digits | 396 | 122 | 71 | 51 | 20 | 24 | 14.78 ms |
| Reverse display | 402 | 126 | 58 | 68 | 25 | 28 | 14.81 ms |
| Swap both | 413 | 132 | 64 | 68 | 23 | 24 | 14.78 ms |

Job 42053 reproduces the direction of the aggregate improvement on a second corpus with one model call. The frozen combined swap adds 5.17 points on item accuracy (paired 95% interval +2.17 to +8.33) and 8.67 points on exact pair accuracy (+4.00 to +13.33). Most of the pair gain comes from support versus insufficient evidence. The combined swap correctly identifies 93 refutations versus 121 with the original prompt and increases false support from 40 to 47 of the 300 non-support items. Macro F1 rises from 61.27% to 66.86%. A higher aggregate score therefore leaves the combined swap unsuitable as an unconditional support gate under the intended false-support constraint. The panel does not estimate population risk or model-training exposure. [Aggregate results](results.json)

Retrieve the pinned [VitaminC test file](https://huggingface.co/datasets/tals/vitaminc/resolve/be6febb761b0b2807687e61e0b5282e459df2fa0/test.jsonl) and the same verified open checkpoint as the NLI4CT comparison. Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.vitaminc_label_transfer TEST_JSONL READER_CHECKPOINT NEW_RESULTS
```
