# Contributor Identity Across Two Publications

Each request specifies two papers and one CRediT role per paper. A verified result links two author ordinals using the same checksum-valid ORCID in the publisher XML. The 32 positive requests require two records for every verified contributor. Neither record alone declares both roles for that contributor. The 27 no-link requests have both roles declared and have published identifiers for all role-eligible authors, with disjoint identifiers under the two requirements.

The builder inspected 130 PLOS ONE XML records and selected 59 requests spanning 26 articles. The input inventory combines the earlier PLOS captures with 12 additional records. The latter came from ORCID searches in [Europe PMC](https://dev.europepmc.org/RestfulWebService). Search results nominate candidate publications. Identity and role evidence come from the article XML, with ORCID checksums validated using [the registry's rule](https://support.orcid.org/hc/en-us/articles/360006897674-Structure-of-the-ORCID-Identifier).

| Identity check | Visible exact / 59 | Visible unsupported links | Hidden-ID unresolved / 32 | Simulated homonym exact / 8 |
| --- | ---: | ---: | ---: | ---: |
| Pair any role holders on the two papers | 3 | 112 | 0 | 0 |
| Match normalized names | 55 | 4 | 0 | 0 |
| Join publisher identifiers | 59 | 0 | 32 | 8 |

The four extra visible name links are missing a published identifier on at least one side. They are unverified under the reference rule. The hidden-ID control removes one side of each verified link and expects an unresolved result. The homonym control deliberately gives an eligible different person the same normalized-name token. Exact agreement for the identifier join is a structural check against labels defined by that join. This replay assumes both papers and roles are given. It does not measure discovery from free text, population-level identity accuracy, or enterprise suitability.

[Source records](sources.json) store paper DOIs, author ordinals, and fixture-local tokens for identifiers and normalized names. [Queries](queries.json) contain declared role pairs and expected ordinal links. [The manifest](manifest.json) pins the input XML inventory and fixture hashes. Data provided by PLOS, with each source credited by its DOI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [CRediT](https://credit.niso.org/) is maintained by NISO.

Replay the frozen fixture with `python -m experiments.credit.cross_source_identity NEW_REPORT.json --replay benchmarks/plos-credit-cross-source-v1`. Rebuild with `python -m experiments.credit.cross_source_identity NEW_FIXTURE XML_DIRECTORY [XML_DIRECTORY...]` using the manifest's publisher XML records.

## Retrieving the two sources

The integrated replay indexes titles and abstracts for all 130 articles. Each of the 59 requests has three views using full titles, middle abstract sentences, or four consecutive normalized title terms. Both source hints and both natural-language role phrases appear in a controlled sentence format. The frozen Qwen3 encoder and synthetic-task ridge head recover the roles. A result counts as exact only when both publications, the resolved status, and all returned author-ordinal links match the fixture.

Compare BM25, dense retrieval, and reciprocal-rank fusion for source selection. The identity-first variants try source pairs from the top four per hint when the initial pair yields no verified link. One BM25 variant can replace only one source. The corroboration variant starts from fusion, allows replacements only from both systems' top three, and locks a source if the two systems agree on its top result.

| Route | Abstract exact / 59 | Short-title exact / 59 | Short-title source replacements | Unsupported short-title links | Short-title P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 59 | 31 | 0 | 0 | 8.40 ms |
| Dense | 39 | 18 | 0 | 0 | 8.47 ms |
| Fusion | 54 | 22 | 0 | 0 | 8.59 ms |
| Fusion and identity-first | 56 | 34 | 13 | 1 | 8.59 ms |
| Fusion and corroborated repair | 54 | 22 | 0 | 0 | 8.55 ms |
| BM25 and identity-first | 59 | 39 | 8 | 0 | 8.44 ms |
| BM25 and one-sided repair | 59 | 35 | 4 | 0 | 8.40 ms |

Every route gets 59/59 exact for full titles. The role proposer gets all role pairs right in these controlled phrasings, so the observed failures are at source selection. The eight repaired BM25 requests represent two publication pairs out of fifteen, and the extra link under fusion comes from a different publication pair than the one requested. The identity signal can repair weak source clues and can also support the wrong papers. These are exploratory shortlist comparisons. Job `40401` motivated testing identity repair on the stronger BM25 baseline, and job `40405` reuses the same cases for that comparison on RTX 5090.

Run [the search replay](../../experiments/credit/cross_source_search.py) with `FIXTURE EXAMPLES_JSON ROLE_TRAINING_JSON PHRASES_JSON EMBEDDING_CHECKPOINT NEW_RESULT_JSON XML_DIRECTORY [XML_DIRECTORY...]`. It verifies all 130 source XML digests. Timings cover warm local request parsing through source-linked output, with no network or index initialization. The source hints are derived from the target articles, and the envelope and role phrases are controlled.
