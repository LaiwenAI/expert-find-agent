# Supervised adaptation of a lightweight evidence verifier

Start from the fixed multilingual DeBERTa checkpoint and the [three-sentence verification study](../scifact-nli-windows-v1/README.md). Split its 309 eligible SciFact training pairs by a stratified paper hash into 233 fit and 76 calibration cases. Keep the 226 development evaluation pairs and their selected lexical windows fixed. Calibration, fitting, and evaluation have disjoint claim IDs and papers. The source corpus has appeared in earlier diagnostic comparisons.

For annotated support or contradiction pairs, train on the shortest complete labeled rationale. For claims with no released evidence, use the lexical window with the benchmark-derived unknown label. This avoids simply assigning the full-abstract verdict to a training window that may omit the labeled evidence. Calibration and evaluation use only lexical windows selected without rationale labels.

Train three epochs with a fixed seed and batch size of eight. Compare the unchanged checkpoint, a trained pooler and classifier (0.59M parameters), and the same head plus the last two encoder blocks (14.77M parameters). Use AdamW with head-only learning rate 0.0003 and last-two learning rate 0.00002. Both trainable runs perform 90 GPU optimizer steps and save their updated tensor subsets. Use the same per-class 5% calibration caps as the earlier verifier study. Recheck raw three-way accuracy on the English and Chinese XNLI test pairs to measure changes outside the adaptation domain.

## RTX 5090 results

| Route | Default support / contradiction / unknown accepted | Calibrated support / contradiction / unknown accepted | English XNLI | Chinese XNLI | SciFact batch P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 40 / 1 / 3 | 13 / 0 / 0 | 88.18% | 81.06% | 19.12 ms |
| Train classifier and pooler | 59 / 9 / 7 | 15 / 0 / 0 | 86.01% | 80.40% | 19.27 ms |
| Train last two encoder blocks and head | 38 / 3 / 4 | 9 / 0 / 0 | 86.51% | 79.68% | 19.41 ms |

The class denominators are 97 true supports, 47 contradictions and 82 unknowns. Changing the default head recovers 19 supports and accepts 12 additional negatives. At calibrated thresholds with zero observed false supports, it retains two more supports than the frozen verifier. The calibration subset is smaller than in the previous study, so the frozen calibrated count here differs from the earlier 20. Training the last two encoder blocks has lower final-epoch training loss and weaker evaluation at the calibrated threshold. Both trained routes lose accuracy on XNLI. This is one training seed and one prior corpus comparison. The experiment does not establish transfer to enterprise requirements or statistical superiority at a fixed risk level. [Aggregate results and weight hashes](results.json)

## Reproduce

Use the pinned mDeBERTa checkpoint, archive and XNLI projection in the [previous multilingual study](../scifact-small-nli-v1/README.md). Run within a Slurm CUDA allocation:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.finetune_nli_verifier \
  --archive-path SCIFACT_ARCHIVE --checkpoint MDEBERTA_CHECKPOINT \
  --xnli-path EN_ZH_JSONL --output NEW_RESULTS --weights NEW_TENSOR_DIRECTORY
```

Job 41264 ran on ws5. Weight subsets use safetensors and are stored with the job results.
