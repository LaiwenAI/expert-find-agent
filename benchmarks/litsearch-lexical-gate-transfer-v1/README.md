# Transfer the PLOS source-expansion gate to LitSearch

Use the [LitSearch benchmark](https://aclanthology.org/2024.emnlp-main.840/) pinned by the repository's existing exact-dense experiments. Search all 64,183 indexed titles and abstracts with BM25 and frozen Qwen3-Embedding-0.6B. Reuse the title-and-abstract vector cache, the 246 author-written ACL and ICLR test questions, and 96 citation-derived calibration questions with disjoint reference IDs and distinct normalized query texts. Some queries can have multiple reference papers.

The narrow policy selects the highest reciprocal-rank sum among papers in the first four results of both retrievers. If the narrow policy abstains, the expanded policy can select a paper in both first-24 rankings, provided it is in the first four on one side. Unrestricted expansion ranks the whole eligible union of agreements and can therefore replace a narrow response. Both selective gates preserve every narrow response and intervene on expansion-only proposals. Both use the expanded candidate's query-to-document IDF coverage. The PLOS route applies the [original frozen threshold](../plos-source-gate-transfer-v1/frozen.json). The local strict comparator freezes the maximal score of any calibration expansion whose paper is absent from the question's reference list. Both use strict greater-than tests. This transfer adapts the PLOS two-source protocol to a single paper and agreement across retrieval methods. It does not test an author join.

Each test question is run with reference papers present and with all known reference IDs masked before ranking. Report recovery of known references and whether a nonreference paper is returned. The latter can be relevant and must not be interpreted as a verified false match. Use query-paired intervals clustered by reference-paper component. The corpus and questions appeared in earlier experiments, but the transferred cutoff was set in a different corpus with no LitSearch labels. The local cutoff uses only the citation-derived calibration questions.

Timing sums live tokenization, query encoding, lexical scoring, exact dense scoring, candidate agreement and any lexical gate check. It excludes indexing, embeddings of documents, serving and evaluation. All procedures run inside a Slurm GPU allocation.

## RTX 5090 results

Job 42446 registers 45 expansion-only candidates when references are present, of which 15 are reference papers. Masking all known references leaves 78 expansion-only candidates. Tests contain 246 paired requests.

| Policy | Known reference returned / 246 | Nonreference return, references present / 246 | Nonreference return, references masked / 246 | Present P95 |
| --- | ---: | ---: | ---: | ---: |
| Top-four agreement | 104 | 72 | 130 | 21.01 ms |
| Expanded agreement | 119 | 102 | 208 | 21.01 ms |
| Frozen PLOS cutoff (0.1121) | 119 | 102 | 208 | 21.02 ms |
| Strict LitSearch calibration (0.6059) | 105 | 75 | 135 | 21.02 ms |

The transferred cutoff lets all 123 expansion-only candidates pass across both views. It preserves all 15 useful expansions when references are present and filters none of the 78 proposals when references are masked. The local strict cutoff scores below the permissive comparison in reference recovery. It blocks 73 of 78 masked expansion proposals while losing 14 of the 15 useful expansions. Relative to top-four, the observed reference recovery gain of unrestricted expansion is 6.10 points (descriptive 95% interval 3.25 to 9.31 points). Locally gated expansion gains 0.41 points (0 to 1.22). The unjudged alternatives require independent relevance annotation. These results show that a fixed lexical score is sensitive to the shift from publisher excerpts to natural literature queries, and offer little support for using the score alone to decide source expansion. [Aggregate results](results.json)

## Reproduce

Reuse the [pinned LitSearch source-withholding projection](../../experiments/litsearch/source_withholding.py) and its [exact dense vector cache](../../experiments/litsearch/dense_cache.py).

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.litsearch.lexical_gate_transfer \
    LITSEARCH_PROJECTION LITSEARCH_DENSE_CACHE QWEN_EMBEDDING FROZEN_PLOS_THRESHOLD NEW_RESULTS
```
