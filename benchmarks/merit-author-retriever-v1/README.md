# Published MERIT retriever on matched LR-Bench comparisons

Use the published MERIT test files with 148 paper-centric
and 829 reviewer-centric pairs. Match each comparison to the
original [LR-Bench release](https://huggingface.co/datasets/Gnociew/LR-bench/tree/6b950b4a6f309d4e6fd3749744d2ddd9005fa6f8)
and public [RATE predictions](https://github.com/Gnociew/RATE-Reviewer-Assignment/tree/8b85915315a5ebca92cda8a0d901ecaaeb8c0cba/predictions).
Both alternatives, the anchor and rating values match.
The released RATE predictions are aligned controls on the
same pairs. The MERIT paper reports a separate RATE-8B run.

Use the [MERIT release](https://github.com/Luli3220/MERIT/tree/5fad0ea06148c3c8817aac841de46c98a75eb40c)
for prefilter and final text formatting. Load the published
[Qwen3-Embedding-8B backbone](https://huggingface.co/Qwen/Qwen3-Embedding-8B/tree/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af)
and [MERIT adapter](https://huggingface.co/Luli3220/MERIT-8B-retriever/tree/6ddce7e8fdfed417d9fd022124529a421560a3f0).
Run the backbone in BF16 with right padding, last-token
pooling and L2 normalization. Prefilter with the unadapted
backbone at 2,048 tokens. Following the published policy,
rank each reviewer's publications for the current target
and keep at most three. The final encodings use a 4,096-token
cap in the local batch implementation.

Compare the backbone and adapter on identical selected
profiles. For two adapter ablations, select the same number
of sources with a frozen 0.6B encoder or use the first
three publications in the release. For a light control,
score the mean of the three best 0.6B publication scores.
All methods use the same pair orientations and labels.
Accuracy is strict, and weighted Kendall loss charges half
the rating gap for a tie. Intervals resample whole anchors.

Encoding times describe batches of distinct strings.
They combine offline and online work as determined by
each profile policy, and cannot be used as service latency.
The benchmark measures familiarity preference on supplied
candidates. It has no expert-specific citation judgments.

## Results

Slurm job `45237` scored every published comparison on the
RTX 5090 without truncation in either 8B stage.

| Route | Paper to reviewer accuracy / loss | Reviewer to paper accuracy / loss |
| --- | ---: | ---: |
| RATE published predictions | 77.70% / 0.1622 | 75.51% / 0.1879 |
| Frozen 0.6B top-three mean | 66.89% / 0.2443 | 76.00% / 0.1857 |
| 8B backbone, author-selected profiles | 71.62% / 0.2080 | 73.46% / 0.2049 |
| MERIT adapter, author-selected profiles | 75.68% / 0.1660 | 76.36% / 0.1790 |
| MERIT adapter, 0.6B prefilter | 74.32% / 0.1775 | 76.96% / 0.1753 |
| MERIT adapter, fixed first three sources | 75.00% / 0.1966 | 73.94% / 0.2012 |

The adapter improves the 8B backbone in both directions with
the same selected profile texts. On paper-centric comparisons,
its accuracy difference from the separately released RATE
predictions is -2.03 percentage points (paired 95% anchor
interval -9.60 to +5.30). The local adapter rerun differs
slightly from the [paper's reported scores](https://aclanthology.org/2026.emnlp-main.2227/).
The execution uses a local BF16 batch implementation.

The compact prefilter uses 9.67 seconds for 2,005 distinct
target and source encodings, versus 46.35 seconds for 1,599
8B prefilter strings. Those counts reflect different cache
inputs. Final adapter encoding takes 109.69 and 109.00 seconds,
respectively. The two prefilters trade quality differently by
direction. The observed means do not identify a universally
safe replacement. [Full aggregates](results.json).

## Reproduce

Use the pinned MERIT test folder, authors' `prepare_topk_LR.py`
and `eval_LR.py`, the RATE prediction directory and the three
verified checkpoints. The local isolated runtime uses
Transformers 4.57.3, PEFT 0.18.0, Accelerate 1.12.0, Hugging
Face Hub 0.36.0 and Tokenizers 0.22.1 with the cluster's
PyTorch installation. Run on a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.merit_author_retriever \
  MERIT_TEST_DIR MERIT_SCRIPT_DIR RATE_DIR   QWEN8B_CHECKPOINT MERIT_ADAPTER QWEN06B_CHECKPOINT NEW_RESULTS --batch-size 8
```

The `--pilot` flag limits the number of rows from each split
and is for execution checks.
