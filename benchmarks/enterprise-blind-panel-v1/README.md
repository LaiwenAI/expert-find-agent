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
| Model queries / human labels for this panel | 0 / 0 |

The panel is large enough for the illustrative 5% risk gate in the [annotation protocol](../../docs/annotation-protocol.md) under independence and sufficiently complete adjudication. With zero events in 123 independent groups, the one-sided 95% binomial upper bound would be 2.41%. This is a sample-adequacy calculation. Grouping catches the specified lexical and issuer keys, and institutional affiliation and semantic duplicates still require review. The availability calculation also depends on the documented exposure manifests.

The local index contains workbook row identifiers and hashes. It contains no copied requirement text or company name. The public summary omits row identifiers and per-group hashes. The previous 20-request feasibility cohort remains a separate panel. [Aggregate audit](results.json)

## Reproduce

Run the [freezing script](../../experiments/freeze_enterprise_panel.py) locally with the original workbook, the original-language cohort directory, the legacy enterprise release directory, the prior-use registry, and a new output directory. Repeat `--exposure` in this order for the same lineage inventory:

1. The protocol files for critic repair, repair pilot, gaps translation, protected translation, segmented translation, original translation, and synthesis, in directory-name order.
2. The real and synthetic exports in the bilingual construction dataset.
3. The demand-first, bilingual-intent, and bilingual retrieval query manifests, in that order.

The stored input hashes verify which artifact versions were read. Keep the index local and freeze the evaluation methods and adjudication protocol before running the panel.
