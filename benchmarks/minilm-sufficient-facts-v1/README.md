# Evidence sufficiency transfer on SufficientFacts

Use the FEVER and HoVer portions of [SufficientFacts](https://aclanthology.org/2022.tacl-1.43/) at [upstream revision 1fbc363](https://github.com/copenlu/sufficient_facts/tree/1fbc36364f5639d294299784958599fda0978472/data/sufficient_facts). Rename `fever_sufficient_facts.jsonl` and `hover_sufficient_facts.jsonl` to `fever.jsonl` and `hover.jsonl`. The runner verifies both digests. Use only the originally supported claims and the released human label after omission. The essential-removal group requires accepting the full view and declining the pruned view. The sufficient-removal control requires accepting both views. The latter includes removals of irrelevant or redundant information. The VitaminC portion of SufficientFacts is outside this transfer panel.

Both views come from the same annotation display, retaining source headings. The runner strips the highlighting from the full view and removes highlighted spans for the pruned view. It normalizes whitespace and checks exact agreement with the normalized released pruned evidence. Neither view exposes removal markup. HoVer can contain several source passages and repeated claims.

Compare the frozen MiniLM and two seeds each of the [SciFact rationale-pair model](../scifact-minilm-window-supervision-v1/README.md) and the [VitaminC strict tail-risk model](../vitaminc-risk-tail-minilm-v1/README.md). Verify saved layer weights against the original reports and keep each model's previously fixed threshold. Report the positive entailment margin (threshold zero) as a sensitivity check. The margin is the entailment logit minus the larger of the neutral and contradiction logits. A pair whose full or pruned view exceeds 512 tokens receives no credit for a correct transition. Paired intervals resample normalized claim clusters, comparing only pairs that fit for both the reference and challenger. The margin-drop AUC measures whether essential removals tend to reduce the margin more than sufficient removals. It uses both-fit pairs and complements the threshold-based results. Timings cover tokenization and a single nonbatched CUDA forward; omit over-limit views.

## RTX 5090 results

The verified run (Slurm 42289) evaluated 698 FEVER and 1,200 HoVer pairs. All views fit the checkpoint. Each cell shows the number of correctly handled pairs, with the group size given in the header. A correct pair requires correctly accepting the original as well as making the correct judgment on the pruned view.

| Reader at its transferred threshold | FEVER essential /519 | FEVER still sufficient /179 | HoVer essential /948 | HoVer still sufficient /252 |
| --- | ---: | ---: | ---: | ---: |
| Frozen | 347 | 151 | 354 | 138 |
| SciFact rationale pair, seed 20261119 | 353 | 150 | 363 | 137 |
| SciFact rationale pair, seed 20261120 | 354 | 149 | 362 | 138 |
| VitaminC risk tail, seed 20261119 | 335 | 112 | 273 | 78 |
| VitaminC risk tail, seed 20261120 | 337 | 114 | 266 | 77 |

At transferred thresholds, the SciFact seeds improved essential-removal point estimates by 6–7 pairs on FEVER and 8–9 on HoVer. Both HoVer paired 95% intervals include zero. With a shared zero margin threshold, the essential-removal result is within three correct pairs of the frozen model for both seeds on either dataset. Raw HoVer false acceptances on essential removals increase from 282 to 371–379. The margin-drop AUC changes from 0.926 to 0.929 on FEVER and from 0.811 to 0.819–0.821 on HoVer.

The VitaminC thresholds declined more essential removals while also declining more original gold evidence and more sufficient controls. On HoVer, pruned false acceptances drop from 290 to 152–160, while acceptance of the paired full evidence drops from 625 to 400–409. At threshold zero their pruned false acceptances instead rise to 735–763. This establishes a deployment risk from threshold transfer and motivates comparing coverage at matched omission risk on independent development data. It does not establish that dropping the decisive passage will reliably trigger abstention for expert search. The public cases supply the gold sources, use induced removals, and carry no expert-level judgments. Pruned-view P95 across readers was 1.17–1.21 ms for FEVER and 1.43–1.46 ms for HoVer. [Aggregate results](results.json)

## Reproduce

Prepare the same [base checkpoint](../scifact-small-nli-v1/checkpoints.json), exact training reports, and saved weights from the two training runs. Execute inside a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.minilm_sufficient_facts \
    DATASET_DIR SCIFACT_REPORT SCIFACT_WEIGHTS VITAMINC_REPORT VITAMINC_WEIGHTS \
    MINILM_CHECKPOINT NEW_RESULTS
```
