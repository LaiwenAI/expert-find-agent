# Rank conjunctions on reconstructed World Bank pages

Use the [n-Clue CrossPage release](https://github.com/sunggukcha/n_clue/tree/8a19258214b2e6973cdc07614908c5caf654fd51)
and its [conjunctive retrieval protocol](https://arxiv.org/abs/2607.24165).
Reconstruct 28 supplied-condition requests (16 two-condition and
12 three-condition) over 82 World Bank PDFs at the recorded
SHA-256 values and page counts. Each request has one released
all-condition document and three known partial documents from
distinct families. A complete label is **grade equal to the
number of conditions**. Scope is the supplied, deliberately
challenging candidate set, without open-corpus recall claims.

The acquisition attempts only HTTPS World Bank URLs present
in the release, capped at 6 MB and 90 pages per document.
It hash-orders eligible requests and candidates, validates bytes,
and keeps requests whose four PDFs have recoverable text. Of 129
attempted source PDFs, 11 downloads fail and two lack sufficient
extractable text or matching page counts. Frozen membership is
reported in the manifest. The replay can reconstruct that exact
membership from the original bytes with `--frozen`. No PDF or
derived page text is included in this repository.

## Protocol

Use physical page numbers from Poppler extraction. Index pages
with at least 120 extracted characters and read up to the first
1,800 characters per indexed page. Report ranking of the known
complete document when all four are scored. Do not use qrels for
ranking, page assignment, or verifier decisions.

- **Whole:** encode the original request and rank by its best page.
- **Independent:** encode listed conditions and rank by the weakest
  among their independent best-page similarities. Reuse is allowed.
- **Distinct:** use the strongest weakest-facet score among
  distinct-page assignments, breaking ties by the sum of scores.
  Top arity pages per condition are sufficient for the best
  assignment because fewer than arity pages can be occupied by
  the other conditions.
- **Distinct body:** repeat distinct assignment after masking
  early pages labeled as contents or containing repeated dotted
  page-number leaders. Keep the original pool when the mask would
  leave fewer than the number of required pages.

Compare these representations with and without the frozen
Qwen3-Reranker-0.6B verifier. The whole verifier reads the top
arity-ranked pages against the original request and uses the
highest margin. The condition variants verify the preselected
page against its specific condition and use the weakest margin.
All verifying arms inspect the same number of condition/page or
request/page pairs. For each verifier, also report top-document
acceptance at fixed margins 0 and −4. These cutoffs are
diagnostics without fitted deployment calibration.

Local P95 covers query encoding, page ranking, selection,
verification when used and serialization. Offline page encoding
is excluded. Document completeness labels do not verify the
exact returned pages; physical citations are available to be
checked separately. The panel is reconstructed native PDF text
without OCR, and the small test uses some shared documents.

## Results

The verified Slurm array `45797` compares all 28 requests on the ws3
RTX 5090. The index holds 3,011 text pages, of which 71 are masked
as potential contents pages in the body route. Every reader
checks 272 query/page or condition/page pairs. The table reports
complete documents ranked first, followed by complete/partial
documents actually issued at the two fixed cutoffs.

| Route | Complete first / 28 | Issued at 0 | Issued at −4 |
| --- | ---: | ---: | ---: |
| Whole dense | 13 | — | — |
| Independent dense | 12 | — | — |
| Distinct dense | 11 | — | — |
| Distinct body dense | 11 | — | — |
| Whole reader | 11 | 8 / 1 | 10 / 1 |
| Independent reader | 13 | 1 / 1 | 1 / 1 |
| Distinct reader | 14 | 1 / 1 | 1 / 1 |
| Distinct body reader | 14 | 6 / 1 | 11 / 1 |

Removing likely contents pages eliminates 13 selected contents
reads in the distinct reader, without changing its total
complete-first count. At the permissive cutoff, the body route
retains 11 complete and one partial result. The whole reader at
the same cutoff returns 10 complete and one partial. These are
small conditional differences. Body verification takes roughly
133 ms at warm P95, versus about 8 ms of whole-dense retrieval
in the reconstructed candidate pool. The expensive route does
not show a strong quality–latency advantage in this comparison.

The page evidence data record which source and physical page
numbers each arm selects. Grades apply to the whole document,
and many requests share candidate documents. Treat the results
as descriptive. The labels are the released CrossPage qrels,
without extrapolating the source's human-audited subset.

[Aggregates and selected page identifiers](results.json).

## Reproduce

Restore the pinned release and use the public membership manifest.
Build with Poppler before running verification in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. python -m experiments.credit.crosspage_pdf_panel \
  CROSSPAGE_V1 PDF_CACHE NEW_PANEL NEW_MANIFEST \
  --frozen benchmarks/crosspage-condition-v1/crosspage-manifest.json

PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.crosspage_condition_search \
  PANEL MANIFEST QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS --order-offset 0
```
