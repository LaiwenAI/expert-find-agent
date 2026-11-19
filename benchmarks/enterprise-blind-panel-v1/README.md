# Candidate panel for blinded enterprise evaluation

This audit derives a metadata-only panel from the 229 sealed demands in the existing original-language corpus. The original cohort has 124 sealed groups. The audit checks the original workbook and both cohort hashes. It treats the construction and validation rows, the entire legacy 100-request cohort, and registered prior-use cases as unavailable for the new panel. It checks 12 additional translation, synthesis, and retrieval manifests for previously selected parent demands.

Group records transitively by the original component, normalized issuer, exact body and full-text aliases, and URL. Issuer normalization removes punctuation and recognized terminal legal suffixes. For pairs involving a candidate sealed row, add lexical links when at least 64 character four-grams match and either Jaccard is at least 0.65 or containment in the shorter text is at least 0.80. Exclude components linked to known exposure. Select one representative per remaining component with a fixed hash order. This selection uses no model confidence, retrieval results or adjudication outcomes.

| Audit stage | Count |
| --- | ---: |
| Original sealed demands / groups | 229 / 124 |
| Sealed demands directly recorded as exposed | 0 |
| Sealed demands transitively linked to exposure | 2 |
| Eligible demands before representative selection | 227 |
| Selected candidate groups | 123 |
| Expert-search requests / human labels for this panel | 0 / 0 |

This panel supports feasibility assessment for the illustrative risk endpoint in the [annotation protocol](../../docs/annotation-protocol.md). With zero events in 123 independent groups, the one-sided 95% binomial upper bound would be 2.41%. This is a sample-adequacy calculation. Grouping catches the specified lexical and issuer keys, and institutional affiliation and semantic duplicates still require review. The availability calculation also depends on the documented exposure manifests.

The local index contains workbook row identifiers and hashes. It contains no copied requirement text or company name. The public summary omits row identifiers and per-group hashes. The previous 20-request feasibility cohort remains a separate panel. [Aggregate audit](results.json)

## Follow-up overlap screening

All 123 representatives were compared with one another and with 2,648 previously used or excluded requests. A frozen Qwen3-Embedding-0.6B encoded title plus leading or trailing demand context on the local GPU. The semantic score is the maximum cosine over the four window pairs. A separate normalized-name screen uses containment and near-string matching. Both screens keep the nearest candidate in each scope.

| Screen and threshold | Within panel | Against prior-use pool |
| --- | ---: | ---: |
| Semantic cosine at least 0.85 | 0 | 4 |
| Issuer similarity at least 0.85 | 0 | 1 |
| Either screen, threshold 0.90 | 0 | 0 |

Counts are panel representatives with at least one nearest match above the diagnostic threshold. The fixed 0.90 review trigger produced an empty form. These screens flag possible overlap and provide no human duplicate, corporate affiliation, or independence judgments. The frozen panel has 123 candidates after the original audit, with no result-based removals. No recommendation or human label was generated. [Aggregate screening](neighbor-screening.json)

## Joint coverage and risk planning

A covered group receives at least one recommendation. An adverse covered group has any promoted expert whose required support or attribution fails review. Count unresolved covered groups as adverse when calculating an upper bound. With independent representative groups, use a one-sided exact lower bound on coverage and an exact upper bound on risk conditional on coverage. Allocate 2.5% error probability to each bound for simultaneous 95% confidence. Select the actual event definitions, margins, and population before examining evaluations. [NIST exact binomial limits](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm) · [NIST simultaneous intervals](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm)

| Illustrative requirement on 123 groups | Observations needed |
| --- | ---: |
| Risk upper bound at most 5%, zero adverse or unresolved | At least 72 covered groups |
| Risk upper bound at most 5%, one adverse or unresolved | At least 110 covered groups |
| Risk upper bound at most 5%, two adverse or unresolved | At least 142 covered groups |
| Coverage lower bound at least 60%, zero adverse or unresolved | At least 85 covered groups for the joint gate |

The third row exceeds this panel's size. At 75% true coverage and 2% conditional adverse risk, the probability of passing both example gates is 14.5% with 123 groups, 69.7% with 360, and 84.0% with 480. This design calculation models one prespecified policy under independent groups. It excludes power for ranking noninferiority, subgroup claims, and correlated organizations. The panel has no adjudicated performance outcomes.

```sh
python -m experiments.enterprise_review_gate 123 --coverage-floor .60 --risk-ceiling .05   --actual-coverage .75 --actual-risk .02
```

## External sampling frames

A university technology-transfer posting supplies four Nantong requests with issuer, project and technical problem fields. The Xi'an science bureau's third and fourth public calls list 50 requests each and link to detailed attachments. These frames may support a separate external cohort after eligibility, organizational overlap, and source availability are checked. The published counts are requests before deduplication. [Nantong posting](https://kjcyc.usts.edu.cn/info/1029/6055.htm) · [Xi'an third call](https://xakj.xa.gov.cn/kjdt/tzgg/2099713781650178049.html) · [Xi'an fourth call](https://xakj.xa.gov.cn/kjdt/tzgg/2105191117492449281.html)

## Reproduce

Run the [freezing script](../../experiments/freeze_enterprise_panel.py) locally with the original workbook, the original-language cohort directory, the legacy enterprise release directory, the prior-use registry, and a new output directory. Repeat `--exposure` in this order for the same lineage inventory:

1. The protocol files for critic repair, repair pilot, gaps translation, protected translation, segmented translation, original translation, and synthesis, in directory-name order.
2. The real and synthetic exports in the bilingual construction dataset.
3. The demand-first, bilingual-intent, and bilingual retrieval query manifests, in that order.

The stored input hashes verify which artifact versions were read. Keep the index local and freeze the evaluation methods and adjudication protocol before running the panel.

For the optional overlap diagnostics, run the following with the same ordered exposure flags. Keep both generated forms and the mappings local.

```sh
PYTHONPATH=src:. python -m experiments.audit_enterprise_panel_neighbors   WORKBOOK COHORT_QUERIES LEGACY_QUERIES REGISTRY PANEL EMBEDDING_CHECKPOINT NEW_OUTPUT   --exposure FIRST_MANIFEST
PYTHONPATH=src:. python -m experiments.audit_enterprise_issuer_aliases   WORKBOOK COHORT_QUERIES LEGACY_QUERIES REGISTRY PANEL NEW_OUTPUT   --exposure FIRST_MANIFEST
```

Use the overlap form when a screen flags candidates. Review the technical intent as same, different or uncertain, and the organizational link as linked, distinct or uncertain. Record the basis for both decisions. Have two reviewers work from separate copies and resolve remaining disagreements independently. Validate the unchanged forms with `python -m experiments.check_enterprise_panel_review PACKET REVIEW_A REVIEW_B`.
