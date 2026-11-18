# Search and credited-clause attribution on a held-out publisher capture

Index publisher titles, abstracts and declarations from the pinned 130-record reference inventory and the [separate PLOS capture](../plos-credit-external-v1/README.md), deduplicated by DOI. The 80 groups from 24 held-out papers supply two requirement versions. Each request has three lines: a citation clue and two task descriptions. The citation clue is an abstract sentence verbatim or the same sentence with every fifth interior word removed at a fixed offset. The hard wording for each requested role comes from the held-out benchmark.

Run the same request with the intended page visible and withheld before selecting the shortlist. Corpus statistics stay fixed. BM25 first-rank attribution is the unguarded control. The other routes use the previous notice-aware ordered score and gap thresholds (0.75 and 0.05) over the top four same-type documents. The thresholds are frozen from the [earlier source audit](../plos-credit-cross-source-v1/README.md#source-confirmation-beyond-literal-matching). On a confirmed publication, compare raw ridge, 4B credited-clause selection followed by ridge, and a direct 4B role decision. All routes require both predicted roles in the same author's declaration. A missing declaration or duplicate predicted role yields an unresolved answer.

## Results

Job `40521` ran on an RTX 5090 with a 165-article resident index. Every route recovers the intended article on all 160 present-source requests per clue style. The abridged-clue results are:

| Route | Exact DOI and author set / 160 | Both versions exact / 80 | Unsupported citations, present | Withheld unresolved / 160 | Withheld unsupported citations | Present / withheld P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 and ridge | 102 | 37 | 28 | 5 | 99 | 8.62 / 8.60 ms |
| Source check and ridge | 102 | 37 | 28 | 160 | 0 | 9.13 / 0.86 ms |
| Source check, 4B clause selection and ridge | 120 | 49 | 19 | 160 | 0 | 24.18 / 0.80 ms |
| Source check and direct 4B role decision | 114 | 44 | 27 | 160 | 0 | 47.76 / 0.79 ms |

Clause selection corrects 19 requests and loses one against guarded ridge. The paired gain is +11.25 points (descriptive 95% article-bootstrap interval +5.71 to +17.07). Against the full reader, it adds six exact results at +3.75 points (−5.71 to +12.50) and roughly halves P95. Source confirmation makes every withheld request unresolved. Unguarded BM25 instead returns 99 unsupported citations and 87 wrong-source empty answers for abridged clues. Literal clues reproduce every present-source output. Their withheld controls show 118 unsupported citations and 70 wrong-source empty answers with unguarded BM25.

Timing includes loading the structured request fields, source retrieval, optional reader execution, and source-bound result assembly from resident data. The source hints come from the target abstracts, the role fields are supplied, and two clue styles reuse the same articles and role scenarios. This experiment measures retrieval and attribution within the controlled publisher task. Open-ended expert ranking, human-written needs, and close semantic source paraphrases require separate evaluation.

## Requirement boundaries in a single paragraph

Render the same two requirements in one sentence joined by “and,” with the quoted abstract clue before or after them. The generated phrases contain their own conjunctions and clauses about other contributors. The parser enumerates possible splits, then uses the first candidate, the maximum sum of two distinct role scores, a constrained 4B choice among candidate splits, or the two original spans as a reference. Every arm then runs the same credited-fragment selection and ridge role mapping. Source confirmation precedes boundary selection, so withholding the publication also avoids unnecessary boundary inference.

On the abridged-clue view, 97 of 160 requests have multiple candidate boundaries. Job `40526` measures:

| Boundary policy | Correct boundary / 160 | Exact DOI and author set / 160 | Both versions exact / 80 | Unsupported citations | Present P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| First conjunction | 92 | 117 | 50 | 16 | 24.41 ms |
| Frozen embedding scores | 114 | 122 | 50 | 13 | 32.58 ms |
| Candidate-constrained 4B | 159 | 120 | 49 | 19 | 38.41 ms |
| Supplied boundaries | 160 | 120 | 49 | 19 | 24.17 ms |

The 4B boundary reader reproduces all author sets from the supplied-boundary reference. Boundary accuracy and final attribution quality differ: improving the segmentation from 114 to 159 correct boundaries does not improve the author sets under the same downstream verifier. Every policy returns unresolved on all 160 withheld requests per clue style, with no selected source. The structured-input regression reruns the existing 640 request views and reproduces all DOI and author outputs from job `40521`.

Add `--prose` to the search command below. This diagnostic learns nothing from its test requests. Its input grammar still uses two fixed envelopes and quoted clues. A separate [source-paraphrase diagnostic](../plos-credit-source-paraphrases-v1/README.md) tests paper identification from unquoted clues. End-to-end attribution from independently written requests remains unmeasured.

## Unquoted source clues with structured role requirements

Job `40554` combines the [24 paper paraphrases and frozen source policies](../plos-credit-source-paraphrases-v1/README.md) with the original 80 requirement pairs. Each selected source goes through the same 4B credited-fragment selection and ridge role mapping. The supplied-source reference abstains whenever the target is withheld.

| Source decision | Correct source / 160 | Exact DOI and author set / 160 | Both versions exact / 80 | Withheld source selected / 160 | Present P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 top hit | 120 | 89 | 37 | 160 | 23.79 ms |
| Dense score and gap | 60 | 43 | 19 | 0 | 30.60 ms |
| Verify one candidate | 120 | 88 | 36 | 0 | 38.31 ms |
| Rerank four candidates | 112 | 83 | 34 | 0 | 52.60 ms |
| Supplied source | 160 | 120 | 49 | 0 | 23.67 ms |

Top-hit verification lowers full request P95 by 27% relative to four-candidate reranking. Its exact attribution difference is +3.13 points (descriptive 95% article-bootstrap interval −12.86 to +19.51). Supplying the correct source yields another 32 exact responses. Unguarded BM25 returns 149 unsupported author citations when the target is withheld. Each guarded method abstains in every withheld view. The 160 cases reuse 24 source clues. Timing covers warm local query parsing, source retrieval and author attribution. Human-written integrated requests and open-world false-confirmation rates remain unmeasured. [Aggregate results](paraphrase-attribution-v1.json)

Append `--paraphrases benchmarks/plos-credit-source-paraphrases-v1/queries.json --source-gates benchmarks/plos-credit-source-paraphrases-v1/gates-results.json --source-reranker RERANKER_CHECKPOINT` to the structured replay command and use a fresh output path.

## Reproduce

Run [the search comparison](../../experiments/credit/search_role_clauses.py) with:

```sh
PYTHONPATH=src:. python -m experiments.credit.search_role_clauses prepare \
  PRIOR_130_RECORD_MANIFEST EXTERNAL_MANIFEST NEW_CATALOG_JSON XML_DIRECTORY [XML_DIRECTORY...]

PYTHONPATH=src:. python -m experiments.credit.search_role_clauses run \
  EXTERNAL_FIXTURE CATALOG_JSON NISO_EXAMPLES_JSON ROLE_TRAINING_JSON \
  EMBEDDING_CHECKPOINT READER_CHECKPOINT NEW_RESULT_JSON
```

Collect the publisher XML from the linked fixtures and run the GPU benchmark in a Slurm allocation. The catalog builder checks source digests against both manifests. Publisher XML and extracted article text are kept outside the source release.
