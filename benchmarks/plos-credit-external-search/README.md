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
