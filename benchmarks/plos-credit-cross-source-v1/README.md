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

## Unavailable source audit

Repeat the same requests with both sources visible, with the left source unavailable, and with the right source unavailable. The availability filter removes the source before candidate ranking and keeps the original BM25 statistics fixed. Include a fourth view with an automatically edited abstract clue to test the limits of literal matching. The required result is unresolved when either requested paper is unavailable, even if a different paper can supply a verified link.

The literal route first looks for a unique exact title, then a unique contiguous match in the title or abstract of visible articles. Both clues need a source match before the identity check can resolve the query. Compare this route with BM25 and the two BM25 identity-repair arms.

| Route | Both visible, exact / 236 | One unavailable, unresolved / 472 | One unavailable, wrong-source links | One unavailable, false empty answers |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 208 | 297 | 1 | 174 |
| BM25 and identity-first | 216 | 284 | 17 | 171 |
| BM25 and one-sided repair | 212 | 284 | 17 | 171 |
| Literal source guard | 177 | 472 | 0 | 0 |

Each wrong-source link in this run occurs on a different request with an unavailable source. A false empty answer confidently returns zero matches based on the wrong papers. One-sided repair behaves identically to unrestricted repair when a paper is withheld, so constraining the number of swaps does not stop this failure. The literal guard accepts all 177 complete requests in the three literal styles, and abstains on all 59 complete requests in the edited-abstract style. The edits are automatic and the results share 15 publication pairs. These controls show the need to verify document relevance separately from author identity and to test source absence apart from ranking on complete collections.

Add `--audit-missing-source` to the search replay. Job `40409` runs the audit on RTX 5090 with the 130-source snapshot and one unchanged embedding checkpoint.

## Source confirmation beyond literal matching

Rank four BM25 candidates for each source hint. Score each candidate against its best matching title or abstract sentence. The coverage verifier measures token overlap with multiplicity. The ordered verifier measures the fraction of hint tokens aligned in order. Both require a minimum score and a lead over the second candidate before contributor identity is checked.

The fixed threshold grid is searched on the 104 indexed articles that supply no test targets. Calibration includes all five clue styles with the intended article present or withheld. The selection maximizes accepted correct sources subject to zero accepted wrong sources in calibration. The test target articles can serve as distractors during calibration. The threshold then stays fixed for both-present and one-unavailable test requests. The abridged clue control deletes every fifth interior-position word according to a fixed offset.

Neither pooled calibration nor separate short and long thresholds finds a nonempty feasible policy. Several correction and retraction titles contain the original article title. With the original withheld, four such long requests reach perfect ordered coverage with a margin of at least 0.5. The document-type control requires the candidate's notice prefix to match the clue's prefix. It recognizes explicit correction, retraction, and expression-of-concern headings. Full reference examples without a notice prefix search ordinary articles. This is a publisher-format cue and needs separate validation on other source conventions.

| Confirming sources | Both present, exact / 295 | Edited exact / 59 | Abridged exact / 59 | Title sketches exact / 59 | One missing, unresolved / 590 | Edited P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Literal source guard | 177 | 0 | 0 | 59 | 590 | 8.52 ms |
| Notice-aware token coverage | 212 | 55 | 55 | 0 | 590 | 9.03 ms |
| Notice-aware ordered matching | 216 | 55 | 55 | 0 | 590 | 9.73 ms |

Both notice-aware policies choose a minimum long-clue score of 0.75. The coverage and ordered margins are 0.30 and 0.05 respectively. Short-clue calibration remains infeasible for both. On these held-target tests neither route accepts a wrong source in either slot, even before the identity join. The coverage rule recovers the same edited and abridged requests at lower P95. The ordered rule answers four additional full-title requests. The input edits reuse article text, and the experiment does not establish coverage on independently written paraphrases.

Add `--confirm-sources` to the cross-source search replay. Jobs `40412`, `40416`, and `40418` explore pooled, length-stratified, and notice-aware calibration on RTX 5090. The notice-prefix rule follows the calibration error analysis. See [the scoring and calibration code](../../experiments/credit/source_confirmation.py).
