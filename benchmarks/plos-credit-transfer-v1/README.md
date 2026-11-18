# Frozen-Route Transfer on New PLOS Sources

This test keeps the role training data, embedding and reader revisions, and all decision thresholds fixed. Its 34 labeled articles are disjoint from the original 100-article catalog. The [frozen policy](frozen-route.json) records the calibration inputs and the thresholds from job `40382`. The [28 transfer phrases](phrases.json) use newly authored task wording with no exact normalized overlap with role training or the 49 NISO examples.

The second [PLOS Search](https://api.plos.org/solr/search-fields/) snapshot has 47 articles and yields a searchable corpus of 44 after removing three overlapping DOIs. Seven remaining articles lack complete author declarations, and three supply no matched role edits. The other 34 supply 128 matched groups. The builder takes up to one per transition stratum per article. Each group supplies two versions for each source lead style. Leads use a title or the middle sentence of an abstract.

[pairs.jsonl](pairs.jsonl) and [manifest.json](manifest.json) define the labeled requests and source provenance. Publisher labels refer to contribution declarations for one article. Both lead styles are source-derived and the new task phrases are controlled authoring, so independent human requests and enterprise relevance remain separate evaluations.

## Results

| Route on authored phrases and abstract leads | Exact author sets / 256 | Unsupported citations | Reader calls | Mean | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 and direct reader | 77 | 84 | 256 | 37.27 ms | 63.01 ms |
| Definitions and reader | 179 | 20 | 256 | 33.05 ms | 36.17 ms |
| Ridge and always read | 234 | 0 | 256 | 32.90 ms | 35.88 ms |
| Ridge without reader | 247 | 1 | 0 | 8.37 ms | 8.66 ms |
| Frozen margin gate | 241 | 0 | 54 | 13.57 ms | 33.41 ms |
| Frozen empty-only gate | 234 | 0 | 237 | 31.13 ms | 35.89 ms |

Job `40385` uses RTX 5090. The margin and agreement gates make the same 202 fast decisions on abstract leads. Timing covers warm local requests through evidence-bound output. All 256 abstract leads recover the intended article. Eight title requests retrieve the same journal correction in place of its original article, so they count as source misses even when the query repeats the title.

### Wording control on the same articles

Job `40389` repeats the 256 abstract requests using [NISO task examples](https://zenodo.org/records/18421449) in the same sentence format. It keeps the article, source lead, target roles, request family and frozen thresholds fixed in every pair.

| Route on NISO wording | Exact author sets / 256 | Unsupported citations | Reader calls | Mean | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 and direct reader | 50 | 32 | 256 | 40.78 ms | 76.32 ms |
| Definitions and reader | 146 | 27 | 256 | 34.19 ms | 37.86 ms |
| Ridge and always read | 172 | 29 | 256 | 34.18 ms | 37.35 ms |
| Ridge without reader | 152 | 35 | 0 | 8.67 ms | 8.84 ms |
| Frozen margin gate | 177 | 24 | 188 | 27.56 ms | 37.41 ms |
| Frozen empty-only gate | 172 | 29 | 251 | 33.81 ms | 37.45 ms |

The fast-route share falls from 202/256 to 68/256 with NISO wording, despite identifying every source under both conditions. Exact task boundaries fall from 205 to 107. The margin gate introduces zero new unsupported citations on these paired samples relative to always reading, and P95 is essentially flat on NISO wording. Report speed and error changes separately for each wording source. The wording control followed the initial transfer and uses the same frozen thresholds.

## Reproduce

The second catalog uses the query below, fields `id,title,abstract`, at most 100 rows, and `publication_date desc,id asc` sorting.

```text
title:("text summarization" OR "named entity recognition" OR "information extraction" OR "topic modeling" OR "document classification") AND journal:"PLOS ONE" AND publication_date:[2022-01-01T00:00:00Z TO 2026-01-01T00:00:00Z]
```

Run [the fixture builder](../../experiments/credit/build_role_transfer.py) with `NEW_SNAPSHOT_DIRECTORY ORIGINAL_CATALOG NEW_FIXTURE --download`. Both catalog captures are checked against the recorded hashes. The [integrated replay](../../experiments/credit/retrieve_and_attribute.py) takes `FIXTURE CATALOG EXAMPLES_JSON EMBEDDING_CHECKPOINT READER_CHECKPOINT NEW_RESULT_JSON`, the original `--role-training` input, and `--frozen-role-route` and `--transfer-phrases` pointing to this fixture's files. It uses the saved margins directly. Add `--transfer-niso-control` for the matched wording control.

Data provided by PLOS. The included XML sources declare [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The manifest credits each source by DOI and records its XML checksum. [CRediT](https://credit.niso.org/) is maintained by NISO.
