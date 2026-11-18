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

## Shared parsing before attribution

The two requirements are now marked in the same request format for every arm. The direct reader decides separately for the two authors in one batch. The factored routes resolve the two task phrases once per request and reuse the predicted roles for both authors. They use nearest-definition embeddings, one batched full-glossary reader, or a reader over the union of lexical and dense top-three roles per phrase. The invariance route attempts to skip that last reader when every distinct role pair in the two shortlists yields the same decisions for both authors.

The two language conditions reuse the same test articles and labels. One uses the 14 authored phrases. The second uses all 49 [NISO task examples](https://zenodo.org/records/18421449), with no exact phrase shared between conditions. These NISO examples were also used in earlier diagnostics. The structured boundary is supplied by the benchmark input; open-ended request segmentation is outside this test.

| Route | Authored groups exact / 112 | NISO groups exact / 112 | NISO request P95 |
| --- | ---: | ---: | ---: |
| Direct reader | 3 | 1 | 24.49 ms |
| Dense definition match | 46 | 23 | 8.40 ms |
| Full-glossary reader | 87 | 44 | 50.94 ms |
| Shortlist reader | 62 | 51 | 37.29 ms |
| Invariance route | 62 | 51 | 36.12 ms |

On NISO wording, the shortlist improves exact group accuracy by 44.64 percentage points over direct reading (paper-bootstrap interval +33.04 to +56.73). It recovers 124/224 exact role sets and makes six false promotions, compared with ten for direct reading. Both correct-role candidates are present in only 144/224 ordered shortlists. Phrasings recur across papers, so the paper bootstrap alone does not quantify generalization to new language.

Invariance skips 40/224 NISO requests and gets 24 of those two-author decisions wrong. The correct role pair is absent from the ordered shortlist in every incorrect skip. Directly relying on agreement within the shortlist therefore repeats upstream coverage errors, even when aggregate accuracy matches the shortlist-reader arm.

Run [the factored replay](../../experiments/credit/factored_role_edits.py) with `FIXTURE EXAMPLES_JSON EMBEDDING_CHECKPOINT READER_CHECKPOINT NEW_RESULT_JSON`. Job `40367` used frozen Qwen3-Embedding-0.6B and Qwen3-4B-Instruct-2507 on RTX 5090. Latencies are warm single-request measurements with one request version and two candidate authors. Source lookup, transport and open-ended segmentation are outside the measured path.

## Reproduce

Run [the builder](../../experiments/credit/counterfactual_role_edits.py) with `SNAPSHOT OUTPUT --capture-from CATALOG PILOT_MANIFEST`. Use the pinned catalog described in [the compound retrieval experiment](../plos-credit-pilot-v1/README.md#compound-roles-with-source-retrieval) and its pilot manifest. The source cache can be reused for subsequent builds. The released fixture contains derived records, with article URLs and hashes for rebuild and audit.

Run [the replay](../../experiments/credit/evaluate_role_edits.py) with `FIXTURE QWEN_CHECKPOINT NEW_RESULT_JSON`. The default split is test. Output includes each group's predicted labels, token counts and measured timings.

Data provided by PLOS. Credit for every source article is recorded by DOI in the manifest. The selected source XML files declare [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and were adapted by selecting and normalizing the structured declarations. [CRediT](https://credit.niso.org/) is maintained by NISO.
