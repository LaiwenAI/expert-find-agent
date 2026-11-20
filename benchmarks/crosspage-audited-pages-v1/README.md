# Locate reviewer-agreed evidence pages at a bounded read budget

Use the [n-Clue adjudication release](https://github.com/sunggukcha/n_clue/tree/8a19258214b2e6973cdc07614908c5caf654fd51/audit/admin/final_adjudication)
to test within-document evidence retrieval. A credited page is a physical,
one-based PDF page selected for the same condition by both reviewer A
and reviewer E. Retain only World Bank v1 cases with recommended positive
support. Exclude the three exceptions flagged by the verification report.
The reconstruction yields 19 agreed positive conditions across 14 source
PDFs and 12 document families. Eight of 22 source PDFs could not be fetched
from their original URLs, accounting for the other nine conditions.
Reconstructed bytes and page counts match the source release. All 19
conditions have a credited page among the 1,087 indexed pages.

## Comparison

The document and individual condition are supplied in every arm. No page
labels are used for ranking, clipping, or acceptance. The dense ranker
encodes the first 1,800 characters of every page. The lexical ranker
searches overlapping 1,800-character windows using document-level inverse
page frequency, exact term coverage and phrase match. The union interleaves
the dense and lexical page rankings, starting with dense and deduplicating.
The frozen Qwen3-Reranker-0.6B checks two or four pages together, reading
either each page prefix or its best lexical window at the same character
cap. It selects the highest-scored page and issues it if its margin exceeds
zero. Pure retrieval always nominates one page.

## Results

All non-timing outputs agree across two rotated runs from Slurm array
45807 on one RTX 5090. P95 is warm local time per supplied condition,
including any live query encoding, lexical selection and reader call.
Offline page encoding and lexical preparation are excluded.

| Route | Credited page in slate | Credited page at rank 1 | Issued credited / other | Reads per condition | P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense only | 9 | 9 | — | 0 | 8.10–8.14 |
| Lexical only | 11 | 11 | — | 0 | 4.21–4.27 |
| Dense 4, prefix | 14 | 12 | 9 / 2 | 4 | 38.12–38.15 |
| Dense 4, focused | 14 | 12 | 9 / 2 | 4 | 40.18–40.29 |
| Union 4, prefix | 17 | 12 | 10 / 2 | 4 | 40.39–40.43 |
| Union 4, focused | 17 | 13 | 10 / 2 | 4 | 40.20–40.29 |
| Dense 2, focused | 12 | 12 | 9 / 1 | 2 | 28.32–28.44 |
| Union 2, focused | 15 | 13 | 9 / 2 | 2 | 26.93–27.50 |

At two reads, the union reaches more credited pages than dense at the same
budget (15 versus 12). It issues the same total number of credited pages
as dense four with a prefix, gaining one condition and losing another, and
reduces P95 by around 28–29%. Union four reaches 17 known pages but issues
only 10 of them. With identical slates, focused reading changes ranked
hits by zero on dense and one on the union, without increasing credited
issuance. Candidate selection and verifier acceptance merit separate
evaluation.

The reference pages are incomplete: an uncredited output might also
support the condition. Issued “other” pages here are unjudged. The panel
contains 19 positive conditions from selected audit cases, with eight
documents shared with the [earlier document-level panel](../crosspage-condition-v1).
These are descriptive, within-document results. The experiment does
not establish false acceptance risk, excerpt entailment, completeness
across all conditions, or end-to-end expert search gains.

[Page IDs and both execution orders](results.json).

## Reproduce

Use the pinned upstream n-Clue checkout, Poppler 26.11.0, the
[public manifest](manifest.json), and revision-matched local model snapshots.
Reconstruction does not redistribute original PDF bytes or extracted text.

```sh
PYTHONPATH=src:. python -m experiments.credit.crosspage_audited_panel \
  N_CLUE_ROOT PDF_CACHE PANEL NEW_MANIFEST \
  --frozen benchmarks/crosspage-audited-pages-v1/manifest.json

# Inside a Slurm CUDA allocation, rerun for offsets 0 and 1.
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.crosspage_audited_search \
  PANEL NEW_MANIFEST QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS --offset 0
```
