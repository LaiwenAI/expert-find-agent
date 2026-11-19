# Small multilingual NLI verifiers for evidence support

Run [multilingual DeBERTa](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) and [multilingual MiniLM](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli) with original evidence as premise and the unmodified claim as hypothesis. Enforce each model's 512-token input limit and explicitly abstain when a complete pair exceeds it. Score entailment as the binary support decision. For the calibrated arms, select an entailment logit margin on the same 309 calibration pairs as the [original verifier comparison](../scifact-criterion-calibration-v1/README.md), using each model's valid inputs and the same separate 5% false-support caps. Freeze thresholds before scoring the same 226 SciFact development pairs.

## SciFact on RTX 5090

| Method | Accepted support / 97 | Accepted contradictions / 47 | Accepted unknowns / 82 | Over-limit pairs / 226 | Batch P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Enterprise reranker | 91 | 41 | 37 | 0 | 154.2 ms |
| Direct Qwen 4B reader | 49 | 4 | 2 | 0 | 798.2 ms |
| Multilingual DeBERTa | 38 | 3 | 7 | 34 | 41.0 ms |
| Multilingual DeBERTa, calibrated | 23 | 0 | 0 | 34 | 41.0 ms |
| Multilingual MiniLM | 41 | 14 | 14 | 29 | 9.6 ms |
| Multilingual MiniLM, calibrated | 8 | 2 | 3 | 29 | 9.6 ms |

Batch time covers tokenization and inference for batches of up to 16. Each method omits its over-limit rows from inference, so the lower times include a work reduction. The table includes all test rows in the class denominators. Calibration uses the same logits and adds no model calls. The zero observed false supports for calibrated DeBERTa come with just 23 of 97 true supports retained. MiniLM is faster and has more false supports at its default decision boundary. These models remain candidates for selective verification or passage-level tests. The results do not establish enterprise support quality. [SciFact aggregates](results.json)

## English and Chinese XNLI

Use all 5,010 aligned test rows from the [official XNLI data](https://github.com/facebookresearch/XNLI), converted to Parquet by the [dataset publisher on Hugging Face](https://huggingface.co/datasets/facebook/xnli). Evaluate the four combinations of premise language and hypothesis language on the same rows with raw three-way argmax. The model cards describe training on XNLI development data. This is a test-split and language diagnostic within the NLI benchmark family, separate from scientific claims and enterprise qualification.

| Model | English / English | English / Chinese | Chinese / English | Chinese / Chinese |
| --- | ---: | ---: | ---: | ---: |
| Multilingual DeBERTa | 88.18% | 82.22% | 83.05% | 81.06% |
| Multilingual MiniLM | 78.90% | 68.94% | 71.48% | 72.32% |

DeBERTa marks 17 of 1,670 contradictions as entailment in the English-only condition and 87 in the Chinese-only condition. Neither model skipped any of the XNLI inputs. The language columns report accuracy over the same translated items. Within-language and mixed-language decisions should be validated separately. [XNLI aggregates](xnli-results.json)

## Reproduce

Use the [checkpoint manifest](checkpoints.json) to fetch the pinned model files. Create each listed model folder under the checkpoint directory and place its manifest entry in the folder as `local-provenance.json`. The runner validates the SHA-256 hashes and label mapping. Fetch the unchanged official SciFact archive from the previous study.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.small_nli_verifier \
  --archive-path SCIFACT_ARCHIVE --checkpoints MODEL_DIRECTORY --output NEW_RESULTS
```

Job 41204 ran the SciFact comparison on ws5. The optional private diagnostic derives reviewer priorities and does not enter the public results.

For XNLI, obtain the pinned [all-languages test Parquet](https://huggingface.co/datasets/facebook/xnli/resolve/refs%2Fconvert%2Fparquet/all_languages/test/0000.parquet). The local preparation step requires PyArrow. The GPU step verifies the projected file digest:

```sh
python -m experiments.scifact.xnli_language_crossing prepare XNLI_TEST_PARQUET EN_ZH_JSONL
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.xnli_language_crossing compare EN_ZH_JSONL MODEL_DIRECTORY NEW_XNLI_RESULTS
```

Job 41221 executed all four language conditions on ws5. Dataset files and private predictions are kept outside this release.
