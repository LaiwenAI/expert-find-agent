# Verify retrieval constraints with literal evidence windows

Use the existing [FollowIR expansion-candidate protocol](../followir-expansion-constraint-gate-v1/README.md) of Core17 and News21. Reproduce the candidate scoring batch used by the preceding experiment, rank each fixed pool by topic-only dense similarity and select positions 5–24. Use one frozen Qwen3-Reranker-0.6B across all four routes. Every route gets the full changed relevance instruction. A positive means the released assessor label under that instruction is positive. Track judge-confirmed negatives, documents invalidated by the instruction change, and unjudged documents separately.

Compare the full document under the reranker's 1,024-token input limit with three views at the same raw 850-character body limit. All short views include the same title. The lead view uses the first characters. Topic and criterion views select a contiguous raw substring with maximum IDF-weighted coverage, using query terms and criterion terms outside the topic, respectively. Each view considers the same offsets at a 425-character stride plus a right-aligned window. Ties favor earlier offsets. The criterion selection never sees a relevance label. The selected text is copied from the source, while the public data supply document-level judgments and no gold annotation of excerpt support.

Score groups of four candidate documents with the official yes/no reranker format. Freeze 5%, 10%, and 20% judged-negative acceptance targets separately for each route using one collection. Test unchanged on the other collection. Report cutoff-free discrimination, accepted positives, judged negatives and invalidated cases, and paired intervals clustered by topic. Report component P95 for selecting and verifying the 20 candidates as one request. It excludes initial retrieval and the candidate-pool query embedding. The full baseline can truncate, and truncation counts are reported.

## RTX 5090 results

Job 42471 reproduces the judged-positive, judged-negative and unknown inventories of the earlier candidate experiment. Core17 and News21 have no shared exact normalized query or source identifier. Gates use the 10% calibration target in the table. Test rates can differ from the target.

| View | Core17 → News21 accepted positive / 131 | Accepted negative / 316 | News21 → Core17 accepted positive / 94 | Accepted negative / 233 | Verify 20 candidates, Core17 / News21 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 23 | 59 | 18 | 8 | 180.02 / 204.06 ms |
| Lead | 17 | 54 | 17 | 9 | 95.68 / 103.38 ms |
| Topic window | 21 | 65 | 20 | 11 | 100.22 / 107.98 ms |
| Criterion window | 19 | 55 | 11 | 12 | 101.33 / 107.49 ms |

The criterion window cuts verification P95 by approximately 44–47% relative to full input. On News21 it retains four fewer positives and accepts four fewer judged negatives. On Core17 it retains seven fewer positives and accepts four more judged negatives. The paired interval for the Core17 positive change is about −13.4 to −1.3 points. In the reverse direction it includes zero. In cutoff-free comparisons, criterion selection changes retained-versus-invalidated AUC from 0.462 to 0.471 on News21 and 0.549 to 0.482 on Core17. Full inputs truncate on three News21 cases, while all bounded inputs fit. The original unaligned replay gave the same qualitative cost and quality pattern.

Short inputs reliably reduce the local verifier workload. Selecting evidence by criterion keywords fails to preserve quality consistently, even against a length-matched lead control. The public judgments cover documents and do not establish whether a quoted substring is sufficient. Retain short windows as a cost control and test an explicit sufficiency decision before asking the system to stop or to read more. [Aggregate results](results.json)

## Reproduce

Use the two pinned FollowIR projections and cached document embeddings from the [previous expansion experiment](../followir-expansion-constraint-gate-v1/README.md), plus the verified embedding and reranker checkpoints.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.followir.constraint_window_verification \
    CORE17_JSON CORE17_VECTORS NEWS21_JSON NEWS21_VECTORS QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```

Run inside a Slurm GPU allocation.
