# Train a lightweight support verifier on claim-matched evidence

Start from the pinned open multilingual MiniLM checkpoint from the [SciFact verifier study](../scifact-small-nli-v1/README.md). Construct 1,200 matched pairs from the official VitaminC training file, split evenly between support–refute and support–insufficient-evidence cases. Within a pair the claim is identical and the evidence differs. The training sampler excludes all pages and case-insensitive claims in the earlier transfer panel and in the [selective-support development and test panels](../vitaminc-selective-support-v1/README.md). Reuse the latter two panels, 400 pairs each, for matched comparison against the prior reader control. These test pages have prior reported baselines.

Compare the frozen checkpoint, supervised adaptation using three-class cross-entropy, and supervised adaptation with an additional pairwise ordering loss. The extra loss is 0.4 times the mean softplus of the non-support minus support example's support margins. A margin is the support logit minus the larger non-support logit. Both trainable arms update the final transformer block and classifier for two epochs under the same seed, pair batches, and optimizer settings. The contrastive term changes training only. Each arm uses a single forward at inference time.

Set each model's thresholds using the same development cases. The first operating point uses the previous original-prompt caps, 21 false supports on refutations and 28 on insufficient-evidence cases. The second caps each type at 10 of 200. Use a strict boundary to reject ties. Evaluate true support and both types of false support separately on the frozen test panel. The development counts are operating-point controls, not guarantees against test distribution shift. Report query–source scoring latency including tokenization, separately by batch size. These are inference timings for given evidence.

## RTX 5090 results

Counts in each acceptance column are true supports / false accepts on refutation / false accepts on insufficient evidence. Denominators are 400 / 200 / 200 on the test panel.

| Model | Three-class accuracy | Uncalibrated | Matched caps | Strict caps | Single-item P95 | Four-item batch P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen MiniLM | 44.88% | 165 / 43 / 11 | 139 / 20 / 6 | 100 / 12 / 2 | 1.25 ms | 1.83 ms |
| Classification adaptation | 59.75% | 305 / 107 / 51 | 144 / 26 / 8 | 98 / 13 / 2 | 1.26 ms | 1.85 ms |
| Contrastive adaptation | 59.25% | 302 / 106 / 45 | 144 / 24 / 6 | 102 / 14 / 2 | 1.27 ms | 1.91 ms |

Job 42190 updated 1.92 million trainable parameters over 150 steps in each trained arm. Both arms improved raw three-class accuracy, with more false supports at the uncalibrated threshold. At the matched caps the contrastive arm accepted five more true supports than frozen MiniLM and made four additional refutation errors. The paired 95% interval for the true-support difference was −1.25 to +4.00 percentage points. At the strict operating point it accepted two more true supports and two more refutation errors. All models exceeded the development cap on test refutations at the strict setting. Threshold decisions differed on up to two of 800 examples when switching from single-item to four-item batches.

The prior Qwen3-4B original one-call control accepted 239 true supports, 17 refutations, and 16 insufficient-evidence cases, with a single-item P95 of 14.88 ms on the same test items in a separate run. The trained MiniLM scores much faster, but the recorded thresholds do not establish a quality-preserving replacement for that control. Results use one seed and supplied evidence. The trained layer tensors were saved on the compute node and are identified by SHA-256 in the aggregate. [Aggregate results](results.json)

Fetch the pinned [training file](https://huggingface.co/datasets/tals/vitaminc/resolve/be6febb761b0b2807687e61e0b5282e459df2fa0/train.jsonl) and the corresponding development and test files. Prepare the projection and train on a Slurm CUDA node.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.scifact.vitaminc_contrastive_minilm prepare TRAIN_JSONL DEV_JSONL TEST_JSONL NEW_PROJECTION
python -m experiments.scifact.vitaminc_contrastive_minilm run \
  NEW_PROJECTION benchmarks/vitaminc-selective-support-v1/results.json MINILM_CHECKPOINT NEW_RESULTS NEW_WEIGHTS_DIR
```
