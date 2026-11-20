# Test constraint-aware rescue with live verification

Use the pinned [FollowIR Core17 and News21](https://aclanthology.org/2025.naacl-long.597/)
projections from the [window verification study](../followir-constraint-window-verification-v1).
Each request has a fixed document pool and document-level judgments
after a change in the relevance instructions. Keep the original
labels for positives, negatives, changed-to-nonrelevant cases and
unjudged documents. Use the same Qwen3-Reranker-0.6B and first
850 body characters with titles. The relevance judgments apply to
documents and do not independently entail the selected excerpts.

Calibrate one fixed score threshold on known negatives among the
first six topic-ranked candidates in a different collection. Target
at most 10% empirical acceptance on that calibration slice. The
threshold transfers unchanged. The tested routes share the same
cutoff. Cross-collection transfer provides a diagnosis on two
existing fixtures and does not guarantee the empirical target.

| Arm | Document choice and verification |
| --- | --- |
| Flat topic six | Score the first six topic-ranked sources in one batch |
| Depth after abstention | Check the first four and, if none is accepted, check the next two |
| Local constraint rescue | Same first four, then rerank topic ranks 5–24 with the instruction and check up to two |
| Full-pool constraint rescue | Same first four, then rerank the remaining supplied pool with the instruction and check up to two |
| Flat instruction six | Use the joint query and instruction embedding from the start and check six in one batch |

The three staged methods share identical first-batch logits and
the same realized verification counts. Report judged positives newly
proposed by rescue, which ones the verifier accepts, and candidates
added past rank 24. The flat instruction arm was added to the second
job as an exploratory alternative after the initial comparison.
Time includes online query embedding, ranking in the supplied pool,
document window selection, and verification. The cached document
embeddings and contents are resident. No corpus-wide retrieval,
service queue, or network cost is measured.

## Results

Job `43577` on one RTX 5090 reproduces the main decisions of
`43573`. Calibration accepts 7 of 77 known negatives on Core17
and 11 of 115 on News21. Each table entry shows accepted positives /
known negatives and warm P95.

| Transfer | Flat topic six | Depth after abstention | Local constraint rescue | Full-pool constraint rescue | Flat instruction six |
| --- | --- | --- | --- | --- | --- |
| Core17 to News21 (32 topics) | 10 / 26 · 36.17 ms | 10 / 21 · 40.02 ms | 10 / 21 · 47.62 ms | 10 / 21 · 47.66 ms | 15 / 27 · 34.05 ms |
| News21 to Core17 (20 topics) | 4 / 3 · 32.97 ms | 3 / 3 · 38.68 ms | 3 / 4 · 45.88 ms | 3 / 3 · 46.01 ms | 4 / 4 · 33.27 ms |

For the three staged arms, realized verification counts are identical
within each transfer (158 and 114). The full-pool policy proposes 11
and 13 judged positives in its rescues. Depth proposes 10 and eight.
All these rescue positives score below or at the transferred
acceptance cutoff. Full-pool rescue uses 11 source positions past
the local depth in each direction, and adds roughly 7–8 ms at P95
against depth alone. Its extra access does not turn into additional
accepted positives. This shows that the verifier and cutoff are a
bottleneck on this reused benchmark slice. It does not identify
whether the short windows omit the decisive evidence or the reader
misinterprets adequate evidence.

The exploratory flat instruction arm obtains five extra judged
positives on News21 at one extra known negative and one extra
unjudged acceptance, compared with flat topic reading. It matches
the positive count and adds one known negative on Core17. Keep
instruction-first retrieval as a baseline before attributing gains to
adaptive expansion. Test the rescue reader on independently annotated
passages before relaxing the transferred threshold.

[Aggregate results](results.json)

```sh
PYTHONPATH=src:. python -m experiments.followir.constraint_rescue_scope \
  CORE17_JSON CORE17_VECTORS NEWS21_JSON NEWS21_VECTORS EMBEDDING RERANKER NEW_RESULTS
```
