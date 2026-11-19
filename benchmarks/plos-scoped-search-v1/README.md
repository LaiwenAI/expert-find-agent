# Retrieve two named evidence scopes before contributor matching

Expand the [controlled publisher attribution test](../plos-credit-source-scoped-v1/README.md) to source discovery. The search index includes a title and the first half of each available abstract. A source description comes from the final two sentences or a bounded set of keywords drawn from those sentences, without the title. The original structured role templates assign each description to a named source. The two descriptions and their mapping to request scopes are supplied, before role interpretation and search. This experiment does not cover discovering the descriptions inside unrestricted prose.

Compare independent dense top-one retrieval, BM25–dense reciprocal rank fusion, a same-ORCID join within the first four fused results on each side, and an expanded join within rank 12 on the alternate side while retaining an anchor within rank four. Rank linked pairs by the sum of fused-rank positions. When the join finds no supported publication pair, leave both-source resolution unfinished. The link graph comes from publisher ORCID fields on the entire available corpus, including distractors. Topic scores use indexed text. No role labels tune the linked-pair selector.

Use the previously evaluated [phrase-only Qwen reader](../plos-scoped-language-v1/README.md) on 120 requirement versions. Interpret each requirement once and replay the same scoped role decisions against every source policy. Measure source retrieval separately on 24 unique paper probes and six unique paired-source targets, then propagate the retrieval outcomes through the role and contributor intersection. Use known-role results as a diagnostic. In a missing-source control, mask either required paper before ranking and count wrong pairs and unsupported identities. This preserves the same query and other candidates.

## Results

The corpus contains 171 papers, including 24 eligible new-source clues. On the 24 individual probes, dense and hybrid Top-1 recover 21 and 20 papers from the result excerpts, and 9 and 12 from keywords. The six paired-source targets share a small number of linked authors.

| Source strategy | Result-clue pairs recovered / 6 | Keyword pairs recovered / 6 | Wrong result-clue pairs accepted with one source absent / 12 | Wrong keyword pairs accepted with one source absent / 12 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 6 | 0 | 12 | 10 |
| Hybrid | 2 | 0 | 12 | 12 |
| Linked top-four | 6 | 2 | 0 | 0 |
| Linked expansion | 6 | 4 | 2 | 0 |

With keywords, linked expansion raises exact scoped plans with correct sources and contributor sets from 15 to 52 over 120 rendered requirement versions compared with linked top-four. With result excerpts, both linked methods reach 61/120. When one result-clue source is absent, expansion emits 13 unsupported contributor selections over repeated requirement versions. Linked top-four emits none. A shared ORCID can recover a weakly ranked report and can also link an incorrect substitute.

Job 41549 used an RTX 5090 on ws5. Shared retrieval P95 was about 9.00 ms across the paired conditions. This timing includes query encoding, score fusion and all four selection arms, and excludes the separately shared role reader. The repeated requirement versions do not add independent source cases. The experiment isolates recovery and forced-ranking risk with supplied descriptions. It leaves unrestricted request parsing, larger publication pools and human expert relevance for separate evaluation. [Aggregate results](results.json)

## Reproduce

Rebuild the projection from the original XML captures used by the [publisher fixture](../plos-credit-source-scoped-v1/README.md) and its reference-exposure directories. The resulting projection has no byline names or literal ORCIDs. The [projection manifest](projection-manifest.json) lists the DOI, publisher XML hash and attribution for every indexed document; abstracts are retained in the local reproducibility projection.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m experiments.credit.scoped_credit_search prepare \
  SCOPED_FIXTURE NEW_PROJECTION XML_DIRECTORY [XML_DIRECTORY ...]
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.scoped_credit_search run \
  SCOPED_FIXTURE NISO_EXAMPLES SCOPED_PROJECTION QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```

The run command uses a Slurm CUDA allocation and a fresh result file.
