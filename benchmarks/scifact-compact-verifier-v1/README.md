# Short evidence contexts for the strong verification path

Keep the same 309 calibration and 226 development pairs as the [live selective verifier](../scifact-live-selective-v1/README.md). Hold the frozen Qwen3-4B reader, claim, wording, and label mapping constant. Compare complete abstracts, the leading three sentences, a lexical window of up to three sentences, and up to two lexical windows. The [window selector](../scifact-nli-windows-v1/README.md) uses claim token overlap. The second window is disjoint, must contribute an uncovered claim token, and is placed in original document order.

Also run live mDeBERTa routing to a full-abstract or three-sentence fallback. Calibrate each Qwen view and each routing policy separately on the same controls with the same per-class 5% false-support caps. The two cascade arms also have a 50% calibration fallback limit. Both learned the same routing interval in this run. All policies keep their calibration settings fixed when changing the batch width. They run fresh model forwards on a single RTX 5090. Alternating route order within each block reduces systematic warm-order bias.

Prior studies explore learned prompt compression and joint rationale/stance modeling. The comparison here controls model weights and limits itself to source-anchored sentence windows. [LongLLMLingua, ACL 2024](https://aclanthology.org/2024.acl-long.91/) · [ARSJoint, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.290/)

## Results

Every column uses the same 97 annotated true supports, 47 contradictions and 82 unknowns. Latency includes window selection, gating, tokenization and inference. The batch width is 16 or 1. Timed work starts from an already supplied claim and abstract.

| Route | Accepted true / false, batch | Batch P95 | Accepted true / false, single | Single P50 / P95 |
| --- | ---: | ---: | ---: | ---: |
| 4B, complete abstract | 48 / 6 | 800.1 ms | 49 / 6 | 27.22 / 39.35 ms |
| 4B, leading three | 23 / 3 | 344.9 ms | 23 / 3 | 16.44 / 20.95 ms |
| 4B, lexical three | 45 / 10 | 352.5 ms | 46 / 10 | 16.68 / 24.12 ms |
| 4B, two lexical windows | 49 / 6 | 576.5 ms | 49 / 6 | 20.82 / 28.20 ms |
| Selective, complete fallback | 39 / 4 | 427.7 ms | 39 / 4 | 5.05 / 40.99 ms |
| Selective, lexical-three fallback | 42 / 5 | 216.4 ms | 43 / 5 | 4.85 / 25.86 ms |

The two-window direct route lowers single-request P95 by 28.3% with identical total accepted supports and false supports in this cohort. It accepts five contradictions and one unknown, compared with four contradictions and two unknowns for the full abstract. Aggregate agreement does not establish case-level equivalence. The three-sentence fallback runs 103 strong-reader inputs serially, as does the complete fallback. Relative to the complete fallback it saves 15.13 ms at P95, accepts four additional true supports, and accepts one additional contradiction.

The complete and complete-fallback routes reproduce the earlier job's batch and single-request acceptance and fallback counts. The public data were reused, and the selected-abstract verification task excludes retrieval and personal attribution. Assess external transfer, error uncertainty and service latency before adopting a single fast path. [Aggregate results](results.json)

## Reproduce

Run within a Slurm CUDA allocation using the same pinned inputs and checkpoints as the linked selective verifier study:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.compact_full_verifier \
  --archive-path SCIFACT_ARCHIVE --short-checkpoint MDEBERTA_CHECKPOINT \
  --full-checkpoint QWEN_4B_CHECKPOINT --output NEW_RESULTS
```

Job 41303 ran both timing modes on ws5.
