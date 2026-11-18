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

## Reproduce

Build with `python -m experiments.credit.compositional_edits NEW_FIXTURE XML_DIRECTORY [XML_DIRECTORY...]`. Replay with `python -m experiments.credit.compositional_edits NEW_REPORT.json --replay FIXTURE`.

Run [the live replay](../../experiments/credit/live_compositional_edits.py) with `FIXTURE NISO_EXAMPLES_JSON ROLE_TRAINING_JSON EMBEDDING_CHECKPOINT READER_CHECKPOINT NEW_RESULT_JSON` on a Slurm CUDA compute node. Use the pinned glossary and training descriptions from the [requirement-edit benchmark](../plos-credit-edits-v1/README.md).
