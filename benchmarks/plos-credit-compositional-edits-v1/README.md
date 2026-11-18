# Cross-publication contributor requirement edits

Each group fixes two PLOS ONE publications and two author ordinals linked by a checksum-valid publisher ORCID. The request contains three role requirements across the two sources. The edit replaces the final role on the second source with a role credited to an identified coauthor. The proposed same-person link then fails one requirement. Where the records support a stable positive control, a different linked author remains eligible for both versions.

| Split | Publication components | Groups | Query views |
| --- | ---: | ---: | ---: |
| Development | 2 | 12 | 96 |
| Calibration | 2 | 12 | 108 |
| Test | 3 | 18 | 156 |

The builder reuses the pinned 130-publisher-record inventory of the [cross-source fixture](../plos-credit-cross-source-v1/README.md) and selects at most six groups per article pair. Linked components share no paper or published ORCID across splits. The 42 selected groups use 14 articles. Stable positive controls occur in 12 groups across two components. These are scoped tests of publisher-declared contributions.

| Evidence intervention | Expected behavior |
| --- | --- |
| Both correct records visible | Support the original request and reject its single-role edit |
| Withhold one record | Mark unresolved unless a visible record already contradicts a hard requirement |
| Substitute a known different coauthor from the second publication | Reject the joined claim |

[Sources](sources.json) provide declarations, ordinals, fixture-local identifiers, and JATS provenance. [Groups](groups.json) retain the match between edits and stable controls. [Queries](queries.json) enumerate the evidence interventions and labels. [The manifest](manifest.json) pins the inputs and fixture hashes. Publisher declarations and metadata are attributed by DOI to PLOS under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Structural replay

| Rule | Correct / 360 | Unsupported promotions | Role checks |
| --- | ---: | ---: | ---: |
| Paper-pooled roles and identity check | 276 | 42 | 828 |
| Author-scoped roles and identity check | 360 | 0 | 828 |
| Author-scoped, stop after a verified contradiction | 360 | 0 | 634 |

Paper pooling wrongly attaches some coauthor roles to a target. The scoped rule implements the label semantics, so its exact replay is a construction check. The ordered rule checks the editable requirement first and cuts role checks by 23%, without measuring model execution or I/O.

## Open-model replay

Job `40474` on RTX 5090 renders the three supplied task spans using pinned NISO phrases. The joint reader receives the target author declarations and explicit publisher identity status. The factored route maps tasks to CRediT labels with frozen Qwen3-Embedding-0.6B and a ridge head trained on 84 existing synthetic descriptions. The program then joins IDs and checks the author's own declaration. The training phrases have no exact normalized overlap with the NISO examples. Both methods receive article and author locations.

| Route | Correct / 156 | Unsupported promotions | Present edit pairs both correct / 18 | Stable pairs both correct / 6 | Warm P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint Qwen3-4B-Instruct-2507 | 73 | 7 | 1 | 2 | 15.11 ms |
| Factored ridge and structured checks | 122 | 5 | 7 | 2 | 8.57 ms |

The factored route labels all 36 known-different-author controls correctly. It still misses 12 of 30 supported requests, and stable-pair accuracy remains low. Test views reuse three publication components and reused taxonomy phrases. Timing measures local decisions given the cited source records and supplied requirement boundaries.

## Decision-aware language checks

All policies reject a known identity mismatch and skip language checks for withheld records. Among the remaining atoms, the ridge head and BM25 glossary retrieval supply sets of possible role labels. Select the smallest shortlist width per retrieval channel that covers every one of the 23 unique visible phrases in the calibration split. The resulting width is fixed for test. The test has 32 visible phrases, 17 of which occur in calibration.

The full-glossary reader resolves every visible atom. The shortlist reader resolves the same atoms using shorter menus. The decision-aware route checks whether every proposed role agrees on membership in that author's declaration. It reads only disputed atoms unless a visible rejection already settles the conjunction. When it reads, it uses the full glossary. Glossary order for each atom is held constant between the full-glossary and decision-aware routes. The route's default decisions depend on proposal coverage.

At width four per channel, the union contains the reference role for all 252 visible test atoms. On the 156 requests, Slurm job `40480` measures:

| Route | Correct / 156 | Unsupported promotions | Supported cases missed / 30 | Atoms sent to reader | Warm P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge top role | 122 | 5 | 12 | 0 | 8.46 ms |
| Full glossary | 126 | 1 | 13 | 252 | 70.09 ms |
| Shortlist glossary | 124 | 2 | 14 | 252 | 49.38 ms |
| Read only disputed atoms | 126 | 1 | 13 | 252 | 77.00 ms |

The decision-aware route reproduces the full reader's output on every request. Each eligible visible atom still has both a declared and an undeclared role among its proposals, so none is removed. The shortlist cuts P95 by about 30% and loses two correct decisions. The prototype with three arms (job `40477`) reproduces the decisions of the shared arms. The initial ridge errors include confusion of Formal analysis with Software and of Methodology with Conceptualization. The test uses the same three publication components and repeated phrases as the preceding replay.

Run [the routing replay](../../experiments/credit/route_compositional_atoms.py) using the live replay arguments below.

## Small model training

The [training replay](../../experiments/credit/train_role_decision.py) uses the same 84 synthetic descriptions. Each run fits on 70 descriptions and selects a checkpoint by accuracy and cross-entropy on 14 separate synthetic descriptions, one per role. NISO exact normalized phrases are excluded from all training and selection. Compare a frozen ridge control, a trained linear head on the frozen encoder, and a head with rank-eight Q/V adapters in the last three Qwen3-Embedding-0.6B blocks. Both learned arms start at the same ridge ranking.

| Training split seed | Ridge correct, atomic / compositional | Trained head correct, atomic / compositional | LoRA correct, atomic / compositional |
| --- | ---: | ---: | ---: |
| 1977 | 36 / 125 | 35 / 124 | 37 / 127 |
| 2027 | 35 / 123 | 36 / 132 | 37 / 128 |
| 2039 | 35 / 124 | 35 / 128 | 33 / 124 |
| 2053 | 34 / 127 | 37 / 128 | 34 / 128 |

Each row evaluates the same 49 NISO phrases and 156 test views. The ridge fitted on all 84 descriptions gets 34/49 and 122/156 in every run. Head-only training uses 14,350 trainable parameters and LoRA uses 137,230. In the original run, LoRA updates all 73,728 initially zero output-projection adapter entries, improves two decisions over the matched 70-description ridge, and makes four unsupported promotions versus five. It does not recover additional supported requests in that comparison (20/30 in both). Sequential warm LoRA P95 across the repeats is 8.65–8.72 ms.

Jobs `40483` and `40487` execute on RTX 5090. The repeat reproduces the original seed's saved checkpoints byte for byte and every decision. LoRA matches or improves the matched ridge on these views but beats the trained head in only one of four runs. These repeats vary training examples and optimization order while reusing the same evaluation cohorts. They measure split sensitivity and leave wording and domain generalization for fresh examples.

Run the training replay with `FIXTURE NISO_EXAMPLES_JSON ROLE_TRAINING_JSON EMBEDDING_CHECKPOINT NEW_OUTPUT_DIRECTORY` and `--seed` set to the chosen split seed.

## Reproduce

Build with `python -m experiments.credit.compositional_edits NEW_FIXTURE XML_DIRECTORY [XML_DIRECTORY...]`. Replay with `python -m experiments.credit.compositional_edits NEW_REPORT.json --replay FIXTURE`.

Run [the live replay](../../experiments/credit/live_compositional_edits.py) with `FIXTURE NISO_EXAMPLES_JSON ROLE_TRAINING_JSON EMBEDDING_CHECKPOINT READER_CHECKPOINT NEW_RESULT_JSON` on a Slurm CUDA compute node. Use the pinned glossary and training descriptions from the [requirement-edit benchmark](../plos-credit-edits-v1/README.md).

## Ranked conjunction of hard constraints

The ranked replay builds fixed candidate orders from visible byline positions and CRediT publisher identifiers. It reveals each expert's declarations one publication at a time. The controller requires a positive witness for every criterion and returns the first permitted witness in canonical source order. A declaration read can resolve several criteria for the same expert. When pessimistic and optimistic assignments to unread witnesses produce the same ranked experts and citations, the output is determined under this monotone rule.

The 108 test views cover 36 requirement versions in each of three availability conditions. They come from 18 edit groups in three publisher-linked components. The original scope requires a specific publication for each role. A separate scope permits either publication from the same identity. Every schedule reproduces the exhaustive ordered experts and citations at Top-1 and Top-3. When either specified publication is withheld, each scoped route returns no confirmed candidate.

Mean expert–publication declarations read with both sources available:

| Role scope and result limit | Exhaustive | Short-circuit in request order | Fixed ordering by constraints per source | Certificate |
| --- | ---: | ---: | ---: | ---: |
| Specified publication, Top-1 | 3.67 | 2.22 | 1.78 | 1.78 |
| Specified publication, Top-3 | 3.67 | 2.67 | 2.17 | 2.17 |
| Either publication, Top-1 | 3.67 | 2.50 | 2.50 | 2.50 |
| Either publication, Top-3 | 3.67 | 3.67 | 3.67 | 3.67 |

Bundled ordering and the certificate tie on this fixture. The comparison validates stopping and citation fidelity on publisher annotations. The experiment observes declaration labels through an oracle, uses supplied identity and role identifiers, and measures work counts with a small number of publication components. Live source discovery, evidence interpretation, and enterprise expertise remain outside this measurement. [Aggregates](constraint-certificate-v1.json)

Run [the ranked replay](../../experiments/credit/constraint_certificate.py) with `FIXTURE NEW_RESULT_JSON`.
