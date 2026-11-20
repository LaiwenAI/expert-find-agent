# Test evidence windows without changing the authors or read budget

Use the fixed five-author shortlist from the
[preceding allocation study](../community-alternate-witness-v1).
Reproduce the Unix & Linux prefix baseline before accepting results.
Evaluate the frozen 195 Unix & Linux and 20 MathOverflow duplicate
questions with open-weight Qwen3-0.6B encoding and verification.
Only answer text is indexed. Use the same live five-pair batch,
the same request length cap and the same positive margin threshold.
The alternative views are selected before scoring, without using
known-answer labels.

| View | Content |
| --- | --- |
| Lead | First 850 characters. |
| Targeted | One 850-character window chosen by corpus-IDF query term coverage. |
| Split | First 420 characters plus a later nonoverlapping window selected for terms absent in the opening. Include a gap marker within the same 850-character cap. |
| Full | Entire available answer, subject to the verifier's 1,024-token packed limit. |

The two fixed-size alternatives fall back to lead unless total
query term coverage by IDF increases. Candidate retrieval, lexical
window selection and live scoring are timed together. Report known
answers issued, the full issuance count, and paired supported P@5
contrasts at each query's smaller common issuance count. Exact
accepted-answer IDs are partial labels. They do not grade other
outputs or confirm the adequacy of each excerpt.

For a diagnostic outside the timed evaluation, replace the duplicate
query with the original question on which the checked known answer
was accepted. Only known-answer rows receive this substitution.
Keep the lead excerpt and five-row batch, and count changes in
verification decisions. This uses the gold association and can
expose wording sensitivity. It cannot prove that a rejection on
the actual duplicate question was incorrect.

## Results

Both tasks of Slurm array `45696` completed on the ws3 RTX 5090
and reproduced all non-timing results.

| View | Unix known / issued | MathOverflow known / issued | Unix warm P95 | MathOverflow warm P95 |
| --- | ---: | ---: | ---: | ---: |
| Lead | 99 / 297 | 13 / 13 | 47.21–47.39 ms | 48.14–48.27 ms |
| Targeted | 100 / 299 | 13 / 13 | 47.50–47.62 ms | 48.58–48.79 ms |
| Split | 102 / 313 | 12 / 12 | 47.99–48.03 ms | 48.74–49.11 ms |
| Full | 104 / 310 | 12 / 13 | 68.31–68.53 ms | 70.34–71.16 ms |

Full changes 363 Unix and 60 MathOverflow views, with 36 and 23
token-limited inputs. All other arms stay within the model cap.
The known-recovery gain from longer context on Unix disappears
at matched issuance. Neither fixed-size alternative shows a positive
matched-issuance difference. Longer context costs about 21 ms
of Unix warm P95 over lead and loses one known MathOverflow return.

Unix has 139 known candidate answers. The lead verifier rejects 40
under the actual wording and accepts 34 of those when given the
original canonical question. One of the 99 actual-wording positives
moves in the opposite direction. MathOverflow has four gains among
seven actual-wording rejections. Use the distinction when designing
a reference dataset. Original wording and displayed evidence should
be reviewed independently before treating such cases as model errors.

[Aggregates for both runs](results.json).

## Reproduce

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.community.first_query_context_repair \
  UNIX MATHOVERFLOW CV_2024 QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/community-alternate-witness-v1/results.json NEW_RESULTS
```

Use the original snapshot receipts and repeat with `--order-offset 1`.
