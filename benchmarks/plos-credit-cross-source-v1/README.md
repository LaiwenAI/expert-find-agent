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
