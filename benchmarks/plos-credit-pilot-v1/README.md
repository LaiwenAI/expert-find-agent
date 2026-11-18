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
