# PLOS CRediT Requirement Edits

Each item fixes one publication and two authors. Both authors have a shared anchor role. The request replaces the second role, keeping the anchor and the evidence. Exactly one author changes label. The matched coauthor keeps the original label under the same edit. The four strata pair either direction of change with either stable label.

| Split | Articles | Matched groups |
| --- | ---: | ---: |
| Train | 28 | 400 |
| Validation | 8 | 96 |
| Test | 8 | 112 |

Each group yields four judgments per rendering. The canonical rendering uses taxonomy labels. The descriptive rendering uses authored task phrases. The redacted rendering removes the target's declaration and changes its reference label to unavailable. The test split has 28 groups in each transition stratum. Split assignment is by DOI. Authors or topics may recur across articles.

[pairs.jsonl](pairs.jsonl) supplies the role edit, article DOI, one-based JATS author ordinals and expected labels. [manifest.json](manifest.json) supplies the author's structured declarations, source hashes and attribution. An omitted role means the supplied article does not declare that contribution. This fixture tests attributable contributions from a known article. Expertise beyond the publication, free-form request equivalence and enterprise suitability require separate judgments.

The builder excludes articles from the earlier [PLOS role pilot](../plos-credit-pilot-v1/README.md), selects 60 DOIs by hash from the pinned search snapshot, and caps each article at four groups per transition stratum. Forty-four articles yield matched groups. Each author's evidence is held fixed across the two versions. Multiple groups reuse articles.

## Test split results

Job `40363` ran frozen Qwen3-4B-Instruct-2507 on an RTX 5090. Every cell shows the number of groups with all four judgments correct. The hybrid uses exact role names when present and checks target evidence availability before invoking the reader. Descriptive requests fall through to the scoped reader.

| Rendering | All-author reader | Target-only reader | Scoped hybrid |
| --- | ---: | ---: | ---: |
| Canonical | 40 / 112 | 49 / 112 | 112 / 112 |
| Descriptive | 8 / 112 | 15 / 112 | 15 / 112 |
| Redacted | 0 / 112 | 0 / 112 | 112 / 112 |

In the descriptive condition, the scoped route gets 25/112 change pairs and 60/112 stable pairs exactly right. It produces 92 false promotions, compared with 32 for the all-author reader. The clustered 95% interval for the scoped reader's change in exact group accuracy is −5.0 to +19.6 percentage points. Scoped descriptive P95 is 36.85 ms, compared with 69.68 ms for all-author evidence. These are warm local batches of four judgments, excluding retrieval. A smaller input does not ensure better attribution. When the target declaration is hidden, the availability guard returns unavailable with zero model calls and zero promotions.

## Reproduce

Run [the builder](../../experiments/credit/counterfactual_role_edits.py) with `SNAPSHOT OUTPUT --capture-from CATALOG PILOT_MANIFEST`. Use the pinned catalog described in [the compound retrieval experiment](../plos-credit-pilot-v1/README.md#compound-roles-with-source-retrieval) and its pilot manifest. The source cache can be reused for subsequent builds. The released fixture contains derived records, with article URLs and hashes for rebuild and audit.

Run [the replay](../../experiments/credit/evaluate_role_edits.py) with `FIXTURE QWEN_CHECKPOINT NEW_RESULT_JSON`. The default split is test. Output includes each group's predicted labels, token counts and measured timings.

Data provided by PLOS. Credit for every source article is recorded by DOI in the manifest. The selected source XML files declare [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and were adapted by selecting and normalizing the structured declarations. [CRediT](https://credit.niso.org/) is maintained by NISO.
