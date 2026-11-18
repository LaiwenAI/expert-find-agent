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

## Language interpretation before attribution

The [NISO role examples](https://doi.org/10.5281/zenodo.18421449) supply 49 human-written tasks covering the fourteen CRediT roles. Each model receives the official role definitions as a glossary. The example text omits the table's target role label. Agreement measures the example's assigned role in the table. Some tasks also plausibly relate to other roles.

| Role mapping | Correct / 49 | Request P95 | Mean reader tokens |
| --- | ---: | ---: | ---: |
| BM25 over the glossary | 19 | 0.06 ms | 0 |
| Frozen Qwen3 embedding | 23 | 7.02 ms | 0 |
| Qwen3-4B with all roles | 30 | 26.35 ms | 410 |
| Qwen3-4B with union of lexical and dense top three | 34 | 22.48 ms | 160 |
| Use agreement, otherwise run the full reader | 30 | 33.33 ms | 285 |

After rotating option positions, the shortlist and full methods score 35/49 and 32/49. Increasing the shortlist to the union of each top five raises candidate coverage from 40 to 45 examples, but yields 22 correct in the original ordering. The wider comparison has one option-order configuration. The timed routes recompute their own glossary retrieval and model inference on a single RTX 5090. Timing uses three warm repeats.

A known-paper replay maps two pre-split phrases to roles and intersects the exact publisher declarations for those roles. It reuses the same 49 descriptions across the 685 derived cases. The full and narrow readers recover 278 and 326 exact role pairs. They return the exact contributor set in 460 and 479 cases and promote an unsupported coauthor in 128 and 89 cases. A correct-looking author set can occur even when the role labels are wrong. These diagnostics measure the propagation of role-mapping mistakes. Integrated compound language, independent expert queries and enterprise transfer require separate evaluation.

Download the pinned NISO PDF, extract it using [the PDF adapter](../../experiments/credit/parse_niso_examples.py), then run [the role router](../../experiments/credit/role_language_router.py) with the extracted examples and original PLOS source snapshot. Job `40303` runs the primary configuration. Job `40306` runs `--option-offset 5` and `--shortlist-size 5` separately. The source package includes the adapters and reports, with upstream NISO text kept in the local input snapshot.

## Joint handling of compound requests

The compound request diagnostic selects one example for each of the 91 unordered CRediT role pairs, with the known source chosen deterministically from the publisher fixture. Each pair combines two NISO task examples into a complete request. A second template adds an explicitly excluded activity. The source paper stays fixed. The examples are authored by NISO and the request templates are generated for this evaluation.

The oracle boundary arms receive the two required task phrases already separated, with any excluded activity removed. The batched control processes both phrases in one embedding batch and one reader batch. The joint full-glossary arm consumes the complete request and predicts two distinct roles with a shared prefix cache. The joint shortlist arm sees the complete request, retrieves a union of lexical/dense roles, and chooses one candidate pair in a single reader forward. All arms intersect the publisher declarations for the predicted roles. Duplicate role predictions are marked unresolved.

| Method | Exact pair, conjunction / excluded | Exact author set, conjunction / excluded | P95, conjunction / excluded |
| --- | ---: | ---: | ---: |
| Oracle boundary, sequential | 49 / 49 | 68 / 68 | 44.11 / 44.21 ms |
| Oracle boundary, batched | 50 / 50 | 69 / 69 | 36.25 / 36.35 ms |
| Joint full glossary | 11 / 20 | 45 / 47 | 40.01 / 40.48 ms |
| Joint shortlisted pair | 10 / 16 | 42 / 48 | 30.30 / 33.20 ms |

Each column of paired counts has 91 requests. Unsupported author promotions occur on 6/91 per template for the batched boundary control and 22/91 per template for the joint shortlist. The shortlist contains both required roles in 38 and 39 cases and correctly selects the pair on 10 and 16 of those respectively. Both shortlist coverage and pair selection limit the method. Count role-pair recovery separately because different role pairs can yield the same author set. These are warm, paired request timings from local input through author-set lookup on a single RTX 5090. The oracle split is a diagnostic control. Its latency excludes producing the split from a complete request.

Constraining model output to a well-formed pair addresses format. Semantic coverage and fidelity of the parsed request require separate checks. [Grammar-Constrained Decoding, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.674/)

Rotating option positions preserves the gap. Across the two templates, the batched boundary arm gets 47 and 47 role pairs right, compared with 9 and 14 for the joint shortlist. The same 91 pairs are reused across both configurations. The strict role check distinguishes correct interpretation from accidental author-set agreement.

Run [the joint request evaluator](../../experiments/credit/joint_role_requests.py) with the NISO and PLOS snapshots. Job `40312` ran the default configuration and `--option-offset 5`. Local outputs are `artifacts/credit-joint-requests-v2` and `artifacts/credit-joint-requests-rotated-v1`.
