# Compare cheap nomination and bounded reading with competing partial matches

Use the pinned n-Clue v1 release and the source bytes already reconstructed
for the [document-level pilot](../crosspage-condition-v1). Exclude its
query IDs, the query IDs in the [audited-page study](../crosspage-audited-pages-v1),
and all 12 document families represented in that audited study. Verify
that the release already includes the published v1 qrel corrections.
Take all remaining requests for which the PDF inventory contains a
released complete match and three partial matches, with four distinct
families in each pool. Pick one eligible complete document and three
partial matches by a deterministic hash order, unrelated to the model
outputs. Documents can be reused across queries and with the earlier
document-level pilot.

The frozen snapshot contains 110 queries (105 two-condition, five
three-condition), 51 World Bank PDFs, and 38 document families.
Complete targets represent 15 families. The most frequent candidate
family appears in 71 queries. This design tests competition within the
supplied four-document pools. The overlap, source type, and
preponderance of two-condition requests limit generalization.

## Comparison

All paths use the same page prefixes and frozen checkpoints. Whole-query
dense and the whole-query reader use the previous pilot definitions.
The condition-wise paths assign a distinct page per condition to maximize
the weakest nomination score, using the sum to break ties. They mask
likely contents pages when enough other pages remain. The sparse path
uses BM25 over page prefixes. The hybrid adds a bounded BM25 contribution
of `0.15 * BM25 / (BM25 + 4)` to cosine similarity for page assignment and
document nomination. The constants are fixed heuristics for this run.

The reader variants score the nominated pages of two or four documents
with Qwen3-Reranker-0.6B. The weakest condition score ranks the verified
documents, which remain ahead of unverified documents. The whole-query
reader ranks all four documents by their strongest whole-query page
margin. No qrel is used in selection, scoring, or acceptance.

Report complete-document rank, number of physical condition/page
checks, and warm serial P95. Report complete/partial output counts when
applying reader cutoffs of zero and −4. Also report a retrospective
comparison over all routes at up to five or ten partial outputs. Each
route admits successive confidence bands, defined by its top document
score, without breaking score ties. This comparison describes the
panel, with no independent threshold calibration. Complete labels
refer to documents, so a complete document can still carry a weak
citation.

## Results

The two rotated runs use Slurm array 45821 on the same RTX 5090 node
and reproduce every non-timing field, including margins and selected pages.
Local timing covers live query encoding, scoring, selection, verification
when enabled, and result serialization. The 1,603-page index and model
initialization are offline. The table reports the observed P95 range.

| Route | Complete first / 110 | Complete in top 2 | Page checks | P95 (ms) | Complete at ≤5 partial | Complete at ≤10 partial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Whole dense | 54 | 78 | 0 | 8.07–8.26 | 9 | 18 |
| Whole reader, all four | 45 | 84 | 900 | 107.47–107.53 | 3 | 10 |
| Condition BM25 | 46 | 77 | 0 | 0.17–0.18 | 9 | 16 |
| Condition dense | 44 | 73 | 0 | 9.88–10.07 | 1 | 2 |
| Condition hybrid | 52 | 79 | 0 | 9.89–10.06 | 1 | 1 |
| BM25 reader, top two | 39 | 77 | 450 | 33.03–33.13 | 3 | 7 |
| Dense reader, top two | 32 | 73 | 450 | 46.48–46.56 | 3 | 4 |
| Dense reader, all four | 23 | 55 | 900 | 107.17–107.21 | 4 | 4 |
| Hybrid reader, top two | 41 | 79 | 450 | 44.04–44.70 | 8 | 8 |
| Hybrid reader, all four | 31 | 60 | 900 | 105.99–106.53 | 9 | 9 |

The hybrid top-two reader improves on condition dense at the same
450-check budget (41 versus 32 complete-first), but it demotes 25 of
the unverified hybrid's correct first choices while recovering 14.
Reading all four increases that loss. At fixed cutoff zero, hybrid
top-two issues 9 complete and 14 partial results, versus 11 complete
and 15 partial for hybrid all-four. Across the retrospective confidence
curves, whole-query dense and condition BM25 give the strongest results
at low partial counts. These measurements support keeping both cheap
baselines and evaluating verification separately from document ranking.

Source mix matters. Full-panel hybrid top-two loses to whole-query dense
by 13 complete-first requests. Removing queries containing the most
frequent family reverses the difference to +6 on the remaining 39.
The paired full-panel result is descriptive and sensitive to the shared
candidate inventory. There is no independent page validation or live
service load in this experiment.

[Aggregate results and selected page IDs](results.json).

## Reproduce

Rebuild both prerequisite panels with their frozen manifests and
the pinned [n-Clue checkout](https://github.com/sunggukcha/n_clue/tree/8a19258214b2e6973cdc07614908c5caf654fd51).
The new [manifest](manifest.json) identifies the exact input panels,
candidate pools, labels, and family IDs. Only the manifest and
experiment outputs are redistributed.

```sh
PYTHONPATH=src:. python -m experiments.credit.crosspage_competing_panel \
  N_CLUE_ROOT PILOT_PANEL AUDITED_PANEL NEW_PANEL NEW_MANIFEST

# Inside a Slurm CUDA allocation, run for offsets 0 and 1.
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.crosspage_competing_search \
  NEW_PANEL NEW_MANIFEST QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS --offset 0
```
