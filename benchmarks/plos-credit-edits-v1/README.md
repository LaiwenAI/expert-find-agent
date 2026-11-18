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

## Retrieval and attribution from plain text

Two sentence templates place a quoted source description before or after the tasks. The tasks have no field labels, and some NISO phrases contain conjunctions of their own. The splitter enumerates candidate boundaries and scores each split by the best pair of distinct glossary roles. Factored arms decode those roles with a batched shortlist reader and return the author ordinals whose own declarations contain both. The dense variant embeds the source lead and all task spans together.

The index contains 100 PLOS articles, with contributor declarations available for 44. The eight test articles each supply title and edited-last-sentence leads. Every arm receives the same requests, including the direct reader that judges all authors from the top BM25 paper in one batch. The oracle split supplies only the task boundary and uses the same BM25 retrieval and role mapping as the factored arm.

| Route | Title author sets exact / 224 | Edited source hits / 224 | Edited author sets exact / 224 | Edited request P95 |
| --- | ---: | ---: | ---: | ---: |
| BM25 and direct reader | 35 | 224 | 35 | 64.50 ms |
| BM25 and factored reader | 124 | 224 | 124 | 38.26 ms |
| Joint dense source and task encoding | 125 | 160 | 92 | 37.81 ms |
| BM25 and oracle task boundaries | 132 | 224 | 132 | 37.12 ms |

Success requires the correct source and exact author set, with a resolved result. The BM25 factored route succeeds on both versions in 32/112 groups, compared with 1/112 for direct reading. Its per-request gain over direct reading is 39.7 percentage points (paper-bootstrap interval +31.3 to +47.3). The splitter recovers 90/224 exact boundaries, and role normalization recovers 103/224 exact role sets. Supplying the oracle split raises these counts to 224 and 113. The dense source variant misses two of the eight edited-description articles, affecting 64 requests. The same leads and task phrases recur across requests.

Run [the integrated replay](../../experiments/credit/retrieve_and_attribute.py) with `FIXTURE CATALOG EXAMPLES_JSON EMBEDDING_CHECKPOINT READER_CHECKPOINT NEW_RESULT_JSON`. Job `40371` uses the same frozen Qwen checkpoints on RTX 5090. Timing covers local parsing, source lookup, inference and construction of source-linked results. The quoted source descriptions are derived from the test articles. This controlled replay leaves independently authored search descriptions, cross-article person identification, concurrent serving and enterprise relevance for separate evaluation.

## Learning role proposals from synthetic tasks

[The training fixture](role-training.json) contains six authored task descriptions per role. None shares an exact normalized string with the 49 NISO examples. Frozen task embeddings form a mean prototype for each role and train a ridge classifier with penalty 0.1. Each method stores 14 vectors and uses one projection at runtime. The source, split selector and lexical glossary remain fixed. The reader sees the same official definitions and only its candidate list changes.

| Role proposal bank | NISO top one / 49 | NISO dense top three / 49 | NISO lexical–dense union / 49 |
| --- | ---: | ---: | ---: |
| Definitions | 23 | 31 | 40 |
| Mean prototypes | 34 | 41 | 45 |
| Ridge classifier | 34 | 43 | 47 |

| Route on edited source leads | Exact role sets / 224 | Exact author sets / 224 | Unsupported promotions | P95 |
| --- | ---: | ---: | ---: | ---: |
| Definitions only | 51 | 74 | 27 | 8.78 ms |
| Prototypes only | 102 | 123 | 42 | 8.82 ms |
| Ridge only | 104 | 124 | 48 | 8.82 ms |
| Definitions and reader | 103 | 124 | 27 | 36.87 ms |
| Prototypes and reader | 120 | 139 | 36 | 36.50 ms |
| Ridge and reader | 131 | 148 | 33 | 36.53 ms |

Ridge proposals with the reader recover both edited versions in 52/112 groups, compared with 32/112 for definitions and reader. The per-request author-set gain is 10.7 percentage points (paper-bootstrap interval +3.1 to +16.5). The learned proposals increase unsupported promotions by six. Dropping the reader increases promotions further, so equal exact-set counts alone would give the wrong deployment decision. All six arms recover the same source on every request and the same 90 exact task boundaries. The atomic examples recur in the compound benchmark and were used in earlier diagnostics.

Add `--role-training benchmarks/plos-credit-edits-v1/role-training.json` to the integrated replay. Job `40375` fits the proposal matrices and runs the full benchmark replay on RTX 5090. The embedding and language model weights remain frozen. See [the trainer](../../experiments/credit/role_prototypes.py).

## Reproduce

Run [the builder](../../experiments/credit/counterfactual_role_edits.py) with `SNAPSHOT OUTPUT --capture-from CATALOG PILOT_MANIFEST`. Use the pinned catalog described in [the compound retrieval experiment](../plos-credit-pilot-v1/README.md#compound-roles-with-source-retrieval) and its pilot manifest. The source cache can be reused for subsequent builds. The released fixture contains derived records, with article URLs and hashes for rebuild and audit.

Run [the replay](../../experiments/credit/evaluate_role_edits.py) with `FIXTURE QWEN_CHECKPOINT NEW_RESULT_JSON`. The default split is test. Output includes each group's predicted labels, token counts and measured timings.

Data provided by PLOS. Credit for every source article is recorded by DOI in the manifest. The selected source XML files declare [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and were adapted by selecting and normalizing the structured declarations. [CRediT](https://credit.niso.org/) is maintained by NISO.
