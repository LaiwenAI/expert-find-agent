# High-recall screening before reviewer evidence assessment

Reuse the 356 fit, 125 calibration, and 129 test cases from the [profile-disjoint MERIT study](../merit-lightweight-decision-v1/README.md). All groups sharing an exact normalized target or reviewer profile signature remain together. Freeze the same lexical and dense decision heads on fit. Add a two-channel control that takes the maximum of standardized dense-top-source and joint-head scores; this skips only if both signals are below a common calibrated floor. Fit the standardization on fit.

For each of the four screens, take the minimum score of the positive pairs in every positive calibration component. Retain candidates at or above the kth smallest component minimum, with k = floor(0.10 × (number of positive calibration components + 1)). Ties are retained. If k = 0, the screen passes every case. The score rule targets missing any positive in a new positive component under exchangeability of component minima. The observed test coverage is reported directly. This is a fixed release with previously used labels and possibly shared authors behind distinct profile signatures, so the 90% level is a calibration target for the study. It does not guarantee an operational recall figure. [Conformal risk control background](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)

All surviving pairs use the same calibrated Qwen3-4B full-history reader; the no-screen control always runs it. Screened pairs count as negatives in final quality metrics. Each arm runs independently on RTX 5090, with alternating arm order. Timings include live query encoding and the reader calls; resident lexical and publication embeddings are precomputed. The binary target is reviewer self-assessment. Bibliographic citation validity is outside this dataset.

## RTX 5090 results

| Screen | Full reads skipped / 129 | Positive pairs retained / 45 | Positive components fully retained / 27 | Final positive F1 | Mean | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 0 | 45 | 27 | 61.39% | 145.7 ms | 218.2 ms |
| Lexical | 9 | 44 | 26 | 62.00% | 135.8 ms | 210.7 ms |
| Best dense source | 38 | 40 | 22 | 62.63% | 111.3 ms | 212.9 ms |
| Dense and lexical head | 11 | 44 | 26 | 61.39% | 139.7 ms | 217.6 ms |
| Two-channel agreement | 32 | 42 | 24 | 61.39% | 118.0 ms | 212.6 ms |

Each cutoff was the second-lowest minimum among 25 positive calibration components. On test, the best-source screen reduced mean latency by 23.6% and P95 by 2.4%. It fully retained 22 of 27 positive components, below the 90% target. Two-channel agreement fully retained 24 of 27. The more conservative lexical and learned-head screens retained 26 of 27 and saved fewer full reads. All screens preserve the 31 true positives found by the full reader in this run. Excluded real positives had already been missed by that reader. A small final F1 increase thus provides no evidence of safe candidate filtering. [Aggregate results](results.json)

## Reproduce

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.decision.merit_recall_screen \
  MERIT_ASSESSOR_TEST_JSON QWEN_EMBEDDING_CHECKPOINT QWEN_READER_CHECKPOINT NEW_RESULTS
```
