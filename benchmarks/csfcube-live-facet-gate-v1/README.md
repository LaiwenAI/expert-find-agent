# Gate live facet verification with cached evidence vectors

Move from the [paired fixed-source diagnostic](../csfcube-paired-facet-v1)
to a live request on all 50 CSFCube query-facet pools. Each pool has
expert-adjudicated paper relevance judgments. Freeze the eight nominees
by fully specified query and full-abstract dense retrieval for every
arm. The evaluation process asserts equal shortlist membership and
order. The nucleus of the request is the query paper and one supplied
facet, not the paper author's individual contribution.

Use the [pinned Qwen checkpoints](../csfcube-lexical-challenger-v1/provenance.json).
Build and deduplicate full and compact evidence vectors offline. The
compact view keeps the title and up to three sentences tagged with the
requested facet, with a lead sentence fallback. The cross-encoder
receives compact query and evidence views on every route that calls it.
Select at most the first two accepted documents in dense order.

| Route | Live verification |
| --- | --- |
| Dense eight | Verify all eight candidates in one batch. |
| Dense four | Verify the first four by dense rank. |
| Facet four | Verify four candidates with the highest compact cosine. |
| Reject only | Reject low compact cosine and verify every remaining candidate. |
| Two-sided | Reject low cosine, accept high cosine, and verify the middle band. |
| Cosine only | Apply the calibrated compact cosine cutoff and skip the cross-encoder. |
| Query reuse variants | Repeat facet four, reject only and cosine only with the already computed full query vector against the compact candidate index. |

All reader decisions use one shared calibration per fold. Fit on
other query-paper folds and exclude articles present as selected
evidence or query papers in the target fold. The reader and standalone
cosine cutoffs each target no more than 10% grade-0 acceptance in
calibration. The low gate retains at least roughly 95% of
calibration grade-2–3 documents accepted by the reader. The high
shortcut targets at most 2% acceptance of grade-0–1 calibration
documents on cosine alone. The reader can accept additional
negatives, so this is not a combined error guarantee. Hold all
thresholds fixed during evaluation.

Each timed route encodes the first query and does exact judged-pool
retrieval. The original cosine routes additionally encode the compact query.
The reuse routes avoid that extra forward, while preserving the
same primary dense shortlist. Full candidate and compact candidate
vectors are indexed offline.
Measure warm wall-clock latency through online shortlist, gating,
verification and source ID serialization. No external document fetch,
network latency or concurrent load is included. Also compare live
decisions with a replay of the full batch scores, to disclose any
observed batch result changes.

## Results

The original six-arm comparison completes twice as Slurm array
`45751`. The final experiment including query reuse completes
twice as array `45756` on the ws3 RTX 5090. The final table shows
warm local P95 rounded to the nearest millisecond.

| Route | Queries with a strong return / 50 | Issued grade 0 / grade 1 | Verified pairs | P95 |
| --- | ---: | ---: | ---: | ---: |
| Dense eight | 26 | 8 / 18 | 400 | 37 ms |
| Dense four | 26 | 8 / 13 | 200 | 24 ms |
| Facet four | 27 | 8 / 19 | 200 | 32 ms |
| Reject only | 26 | 8 / 18 | 286 | 42 ms |
| Two-sided | 26 | 8 / 18 | 270 | 42 ms |
| Cosine only | 24 | 7 / 19 | 0 | 17 ms |
| Reuse query, facet four | 27 | 8 / 16 | 200 | 24 ms |
| Reuse query, reject only | 27 | 7 / 16 | 305 | 35 ms |
| Reuse query, cosine only | 27 | 7 / 15 | 0 | 9 ms |

The original negative gate verifies 28.5% fewer pairs than dense
eight but takes longer. Reuse removes the second query encoding.
With a four-read budget, reusing the primary vector makes
facet selection as fast as dense four. Its extra strong return
comes with more grade-1 issuance. The unverified cosine shortcut
returns strong sources on seven queries missed by dense eight,
and loses six that dense eight returns. Its net change is +2 points,
with a paired 95% interval of −13.04 to +17.31 points. Thus the
fast path is a candidate for selective verification, while the
trial does not establish quality noninferiority or suitability
to issue evidence without direct checks.

The index has 4,173 full and 5,968 compact vectors, about
41.5 MB in vector arrays. Identical shared shortlist outputs
are observed across all routes. Partial reader batches can
change the final choices compared with replaying full-batch
scores. The result file records the differences and reports
each live run. Keep dense four as the efficiency comparator.

[Results for both final runs](results.json).

## Reproduce

Use the same pinned source files and weights as the paired test.
Run twice in a Slurm CUDA allocation, rotating with the order offset.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.csfcube.live_facet_gate \
  CSFCUBE_DIR benchmarks/csfcube-lexical-challenger-v1/provenance.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS --order-offset 0
```
