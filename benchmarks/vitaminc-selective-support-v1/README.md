# Calibrate support-only decisions with a selective second check

The [label-order transfer study](../vitaminc-label-transfer-v1/README.md) found improved three-class accuracy and increased false support under the combined swap. This study freezes both prompts and calibrates support as a binary acceptance decision. Any result that does not meet the support gate is left unaccepted. This task makes no separate decision between refutation and insufficient evidence.

Sample 400 support–non-support pairs from the pinned VitaminC development set and 400 from the test set. Each split has 200 support–refute pairs and 200 support–insufficient-evidence pairs. Keep each pair's claim fixed, with distinct evidence contexts. Use one pair per page and exclude the pages and case-insensitive claim strings used in the earlier transfer panel. The new development and test panels also share neither a page nor a claim string.

A request first uses either the original or combined-swap prompt. Its support margin is the support logit minus the larger of the two non-support logits. A direct gate applies one threshold. The selective gate accepts above an upper threshold, leaves below a lower threshold unsupported, and consults the other prompt in between. Accept in that region only if the second margin meets its threshold. For each policy, choose thresholds on the development set to maximize true support. Require both false-support counts to be at most their counts under the uncalibrated original prompt and require a second call on at most 25% of development items. Ties prefer fewer second calls and then fewer false accepts. Freeze every selected threshold before assessing the test panel.

Record both live Qwen3-4B forwards per item and alternate which runs first. Factored per-item latency sums the individually measured model-path times only when the policy requests both. It includes prompt construction, tokenization and inference, and excludes request serving. Resample whole pairs for accuracy differences with the original uncalibrated prompt. The development error caps are operating-point criteria and impose no test-set guarantee.

## RTX 5090 results

| Policy | True supports / 400 | False accepts on refutations / 200 | False accepts on insufficient evidence / 200 | Both support decisions right / 400 | Second calls / 800 | Factored P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, zero margin | 239 | 17 | 16 | 209 | 0 | 14.88 ms |
| Combined swap, zero margin | 290 | 21 | 17 | 256 | 0 | 14.87 ms |
| Original, calibrated | 239 | 17 | 16 | 209 | 0 | 14.88 ms |
| Combined swap, calibrated | 228 | 11 | 10 | 208 | 0 | 14.87 ms |
| Selective, calibrated | 239 | 17 | 16 | 209 | 0 | 14.88 ms |

Job 42069 fits and freezes the policy under development caps of 21 false accepts on refutations and 28 on insufficient evidence. The selective family includes direct gates. It chooses the uncalibrated original rule because no eligible second-call gate beats it on the development objective and ties prefer fewer calls. The combined swap at zero margin accepts 51 additional true supports on test alongside five additional false accepts. The calibrated swap uses a support margin of 10. It removes 12 false accepts in total relative to the original, with 11 fewer true supports. The paired 95% interval for its true-support rate change is −5.75 to +0.25 percentage points. The corresponding false-support intervals are −5.50 to −1.00 points on refutations and −6.00 to −0.50 points on insufficient evidence.

Within this threshold family and calibration budget, the prompts offer different operating points and the selective rule does not expand the observed frontier on the chosen development objective. Test differences are diagnostic and do not establish a risk guarantee. [Aggregate results](results.json)

Retrieve the pinned [development file](https://huggingface.co/datasets/tals/vitaminc/resolve/be6febb761b0b2807687e61e0b5282e459df2fa0/dev.jsonl) and the test file and verified reader checkpoint from the transfer study. Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.vitaminc_selective_support DEV_JSONL TEST_JSONL READER_CHECKPOINT NEW_RESULTS
```
