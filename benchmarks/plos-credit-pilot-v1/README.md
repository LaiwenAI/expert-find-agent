# PLOS CRediT Role Pilot

Data provided by PLOS. Each original article and its authors are credited through the DOI links in [the manifest](manifest.json). All 14 source XML records declare [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This adaptation selects and normalizes the structured author-role declarations. The [CRediT role vocabulary](https://credit.niso.org/) is maintained by NISO.

The pilot uses 14 PLOS ONE articles returned by a frozen title query about machine translation, information retrieval, and recommendation systems. Twelve articles contribute 132 paper-and-role queries. Each query contrasts authors with a declared role against coauthors without that declaration. There are 244 positive author-role labels and 385 same-paper role absences.

[queries.json](queries.json) uses one-based JATS author ordinals. The DOI and source field identify the attributable record. An undeclared role leaves the author's experience and ability unknown. These samples evaluate published contributor attribution on known papers. Enterprise suitability and independent adjudication require separate evidence.

Rebuild with `python -m experiments.credit.plos_role_inventory --output <new-directory>`. The live query can change as publisher data change. The manifest pins the captured search result and each XML digest. To rebuild from that exact snapshot, pass `--source-cache <snapshot-directory>` containing `search.json` and the corresponding XML files. The released fixture includes the derived labels and the manifest.

## Attribution controls

Slurm job `40273` tests 629 author-role decisions from 132 queries across 12 articles on an RTX 5090. It compares coauthorship as a proxy, deterministic lookup in the visible publisher declaration, and the fixed open-weight Qwen3-4B-Instruct-2507 reader. Three choices distinguish an explicitly listed role, a role omitted from a visible complete declaration, and an unavailable target declaration. Other authors' roles supply same-paper distractors. The four prompt variants keep the same byline and abstract. They show all declarations, only the target declaration, all declarations except the target, or no declarations.

| Evidence visible | Model accuracy | False role promotions | Unsupported absence claims | Mean input tokens |
| --- | ---: | ---: | ---: | ---: |
| All declarations | 93.96% | 17 / 385 | — | 518 |
| Target declaration only | 98.41% | 9 / 385 | — | 414 |
| Other authors' declarations only | 0.00% | 67 / 629 | 562 / 629 | 496 |
| Byline and abstract only | 3.34% | 3 / 629 | 605 / 629 | 392 |

Scoping to the target declaration recovers the exact positive-author set for 123/132 queries, compared with 107/132 for all declarations. The paired per-decision accuracy difference is +4.45 percentage points (95% article-bootstrap interval +0.31 to +8.74). Warm batch P95 over batches of 16 falls from 698 to 476 ms. Timing covers warm local tokenization and model classification.

The byline proxy promotes all 629 candidates and coincides with 244 declared roles. Structured lookup agrees with all 629 published labels when the target record is visible and returns unavailable when the target record is hidden. This control measures agreement with the publisher tags. Independent adjudication and enterprise transfer remain untested.

Job `40279` repeats the same cases with cyclic permutations of the three option meanings. Scoped accuracy remains higher than the all-author input (97.93% versus 88.08%). When other authors' declarations remain and the target record is removed, the reader falsely attributes the role in 213/629 cases, claims absence in 405/629, and correctly reports unavailable in 11/629. The failure persists under option rotation. These checks motivate deciding whether the target record is available before prompting the model.

Run [the attribution evaluator](../../experiments/credit/attribution_controls.py) on the pinned PLOS XML snapshot and the Qwen checkpoint. Run with `--rotate-labels` for the option-order control. Local outputs are `artifacts/plos-credit-attribution-v1` and `artifacts/plos-credit-attribution-rotated-v1`.

## Compound roles with source retrieval

Slurm job `40297` retrieves a paper before checking its contributor declarations. The index contains 100 PLOS papers. The test has twelve labeled papers with four controlled query views each. These views use the title, an abstract body excerpt, the final abstract sentence, and a light edit of that sentence. They are derived from the source records and share the same twelve papers. The controls compare title-and-introduction and full-abstract indices. The dense embedding truncates at 512 tokens.

| Source policy | Correct source / 48 | Mixed request P50 / P95 |
| --- | ---: | ---: |
| Full-abstract BM25 | 48 | 0.097 / 0.184 ms |
| Full-abstract dense | 44 | 6.991 / 7.119 ms |
| Exact title, then short-index lexical/dense agreement | 33 | 7.110 / 7.322 ms |
| Exact title or unique excerpt, then full-abstract BM25 | 48 | 0.040 / 0.178 ms |

A normalized quote is accepted only when it occurs in one abstract in the snapshot. The combined policy uses 12 title hits, 24 literal excerpt hits and 12 lexical fallbacks. The exact route is faster on literal inputs. Edited excerpts miss the literal route and add overhead relative to full BM25 (P95 0.190 versus 0.118 ms). Timing on ws5 covers a local lookup from the query lead through a role check, with warm resident indices and seven repeats. It is a controlled source-recovery workload. Natural paraphrases, broader corpora, concurrent service tails and latency outside the process require separate tests.

The 132 atomic role queries yield 685 role pairs, with 300 having no single author declared for both roles. Given the correct paper, same-author intersection recovers all 685 exact sets against the publisher tags. Matching either role recovers 137 and promotes an unsupported coauthor in 548. Paper-level pooling promotes an unsupported coauthor in all 685 selected contrasts. With the correct paper retrieved but its role records withheld, the same-author route returns unknown on all 685 cases. For edited excerpts, short-index agreement leaves 383/685 unresolved. Full-abstract BM25 and the combined policy both recover all 685 exact sets on those excerpts. The controls distinguish source lookup failure from wrong-author attribution.

Run [the compound retrieval evaluator](../../experiments/credit/conjunctive_source_retrieval.py) with the pinned XML sources and a [PLOS Search API](https://api.plos.org/solr/search-fields/) snapshot. The snapshot requests `id,title,abstract`, 500 rows, sorted by `publication_date desc,id asc`. Its query is `title:("machine translation" OR "information retrieval" OR "recommendation system" OR "recommender system" OR "transformer model" OR "text classification" OR "sentiment analysis") AND journal:"PLOS ONE" AND publication_date:[2020-01-01T00:00:00Z TO 2026-01-01T00:00:00Z]`. The evaluator pins the returned JSON hash. Local result: `artifacts/plos-credit-conjunctive-v5`.
