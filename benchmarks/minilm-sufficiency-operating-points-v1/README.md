# Evidence retention at common omission risk

Replay the [human-judged SufficientFacts study](../minilm-sufficient-facts-v1/README.md) with the same open MiniLM checkpoint and four saved parameter subsets. Verify all original decision counts before recalibrating. Use no new training. The original view was labeled supported by the source dataset. The pruned view has a released human sufficiency label. A false acceptance is a support decision on a pruned view labeled insufficient.

In each direction, calibrate separately for each reader on one dataset and evaluate the fixed threshold on the other. Exclude target rows whose normalized claim occurs in calibration. Set the threshold to retain as much evidence as possible subject to a 5%, 10%, or 20% empirical false-acceptance cap on calibration. Accept only margins strictly above the threshold. The shared caps are calibration targets. A different target distribution can produce different observed false-acceptance rates.

As a diagnostic of score separation, also calculate each reader's retention at the same risk cap on the target itself. These retrospective curves use target labels and cannot estimate a deployed threshold's performance. Report retention of the original evidence, retention of pruned views still judged sufficient, and decisions correct in both views. Paired intervals condition on the fitted thresholds and resample normalized claim clusters. The two datasets and thresholds are repeated views of the same benchmark, and shared Wikipedia sources can remain across datasets.

## RTX 5090 results

Slurm job 42297 reproduced prior decisions across all five readers. No exact normalized claim overlap occurred between FEVER and HoVer. Use the 10% calibration target to illustrate the result. Counts in the table give pruned false acceptances, full supports retained, and still-sufficient pruned controls retained. Denominators are 948 / 1,200 / 252 for HoVer and 519 / 698 / 179 for FEVER. Ranges span the two trained seeds.

| Calibration → evaluation | Reader | Pruned false acceptances | Full supports retained | Sufficient controls retained |
| --- | --- | ---: | ---: | ---: |
| FEVER → HoVer | Frozen | 93 | 351 | 61 |
| FEVER → HoVer | SciFact rationale pair | 96 | 383 | 67 |
| FEVER → HoVer | VitaminC risk tail | 164–170 | 506–523 | 90–92 |
| HoVer → FEVER | Frozen | 52 | 502 | 124 |
| HoVer → FEVER | SciFact rationale pair | 51 | 512–513 | 125 |
| HoVer → FEVER | VitaminC risk tail | 35–38 | 440–465 | 111–116 |

Retrospectively setting the same cap on HoVer produces 94 false acceptances for every reader. The SciFact models retain 382–383 full supports versus 358 frozen, a paired gain of 2.0–2.1 percentage points. The seed-specific 95% intervals are approximately [0.58, 3.52] and [0.58, 3.64] points. Retention of sufficient pruned evidence moves from 61 to 66–67, with paired lower bounds at or below zero. The VitaminC models retain 373–380 full supports, and their paired intervals both include zero.

At the tighter 5% HoVer cap, the SciFact readers retain 198–199 full supports versus 207 frozen. At 20%, the difference narrows and the paired intervals include zero. At the 5% FEVER cap the SciFact readers retain 414–418 full supports versus 402 frozen. The full-support gain is therefore most consistent around the 10% operating point in this study. It remains small and gets weaker when the remaining evidence has already been shortened. Transferred thresholds can miss the target materially. At a 10% FEVER calibration target, the VitaminC readers accept 17.30–17.93% of HoVer omissions. These are public gold-source experiments. They motivate validating a fixed gate on an independent target-domain panel and testing whether any gain survives open retrieval and expert-level judgments. [Aggregate results](results.json)

## Reproduce

Prepare the pinned dataset, training reports, CUDA checkpoint and verified weights as in the preceding SufficientFacts study. Supply its aggregate JSON to verify the replay. Execute inside a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.minilm_sufficiency_operating_points \
    DATASET_DIR SCIFACT_REPORT SCIFACT_WEIGHTS VITAMINC_REPORT VITAMINC_WEIGHTS \
    MINILM_CHECKPOINT PREVIOUS_RESULTS NEW_RESULTS
```
