# Selective short-to-full reading at a fixed candidate budget

Use the aligned [FollowIR evidence-window cohort](../followir-constraint-window-verification-v1/README.md). Freeze the same 20 candidates at topic ranks 5–24 and partition them into five blocks of four in the original order. Use the same Qwen3-Reranker-0.6B, changed instruction, short lead view and full input. The candidate labels are document-level assessor judgments. Separate previously supported but subsequently invalidated documents and unjudged cases.

Compare four policies. Short-only scores all candidates using their first 850 body characters and titles. Full-only scores every full candidate. The fixed escalation control scores short views, then reads the first two blocks in full. The adaptive route scores short views and ranks blocks by the smallest absolute margin distance from the short acceptance cutoff. It reads at most two blocks, provided they are within a calibrated radius. Full judgments supersede short judgments on those blocks.

Set short and full acceptance cutoffs to at most 10% judged-negative acceptance on the calibration collection. Select the adaptive radius from the calibration distance quantiles and the no-escalation control. Among settings within the calibration risk target, maximize retained judged positives, then prefer fewer full blocks. The fixed control shares the same maximum escalation budget. Its calibration risk is recorded separately. Transfer the complete policy from Core17 to News21 and in the reverse direction. A calibration target does not guarantee the same held-out risk.

Measure each route live after calibration. Full escalation reuses the original block composition to preserve numerical behavior. Verify that all live decisions and selected blocks reproduce the offline traces. Timings include verifier tokenization, the selected model calls and local policy logic. The candidate pool, document text and short views are already resident. The timings exclude query encoding, initial retrieval, window creation, network transport and storage. They are component results for groups of 20.

## RTX 5090 results

Job 42478 reproduces the previous short-lead and full acceptance counts in both directions. All live block selections and decisions match the offline score traces. Counts below are accepted judged positives and negatives, followed by full-read blocks and live component P95.

| Transfer | Route | Positives | Negatives | Full-read blocks | P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Core17 → News21 | Short | 17/131 | 54/316 | 0 | 104.28 ms |
| Core17 → News21 | Full | 23/131 | 59/316 | 160 | 205.91 ms |
| Core17 → News21 | Rank first | 21/131 | 55/316 | 64 | 182.37 ms |
| Core17 → News21 | Margin route | 19/131 | 55/316 | 20 | 162.24 ms |
| News21 → Core17 | Short | 17/94 | 9/233 | 0 | 98.50 ms |
| News21 → Core17 | Full | 18/94 | 8/233 | 100 | 185.33 ms |
| News21 → Core17 | Rank first | 18/94 | 8/233 | 40 | 171.21 ms |
| News21 → Core17 | Margin route | 15/94 | 9/233 | 31 | 171.31 ms |

In News21, the margin route costs 57.96 ms more at P95 than short-only and retains two more judged positives, alongside one additional judged negative. Compared with full-only, it loses four positives, with a paired interval of −5.80 to −0.71 percentage points. In Core17 it retains fewer positives even than short-only, despite 31 escalation blocks. Rank-first reproduces the aggregate full acceptance counts in Core17, although the paired outcomes vary and the calibration risk was 33/316 rather than the 10% target. It also retains more positives than the margin route in News21 at the same observed negative count and higher cost.

A small short margin on its own is weak evidence that reading the long version will improve the decision. The measured route should remain a control. The setting uses document-level labels, fixed pools and already available text. It does not determine whether a quoted passage meets an enterprise constraint, or whether reads from production storage make escalation worthwhile. [Aggregate results](results.json)

## Reproduce

Use the pinned [FollowIR inputs and embedding caches](../followir-constraint-window-verification-v1/README.md#reproduce) and verified embedding and reranker checkpoints. Run inside a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.followir.selective_window_escalation \
    CORE17_JSON CORE17_VECTORS NEWS21_JSON NEWS21_VECTORS QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
