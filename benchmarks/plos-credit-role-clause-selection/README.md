# Separating the credited task before role mapping

The [earlier boundary test](../plos-credit-role-transfer-hard-v1/README.md) and the [separate publisher capture](../plos-credit-external-v1/README.md) use role descriptions that mention another person's contribution. This comparison uses the same article evidence and labels, fixes the previous 84-description ridge head, and varies only how the task is represented at inference.

At a semicolon or one of four explicit contrast markers, split a phrase into two contiguous fragments. The fixed first and last policies use position alone. Frozen Qwen3-Reranker-0.6B judges which fragment represents the credited action. A second selector uses Qwen3-4B-Instruct-2507 to choose the first fragment, second fragment, or full statement. The learned selectors alternate candidate ordering by a stable text hash. The selected fragment goes through the same ridge head. Compare these methods with raw ridge and a direct full-glossary 4B decision. Each arm receives provided requirement boundaries and article identity.

## Results

Job `40506` runs the original phrase banks on an RTX 5090. The parser splits 20 of 28 earlier boundary phrases, 21 of 28 external boundary phrases, and zero straightforward control phrases. Job `40511` mechanically exchanges segments around the first semicolon in the external bank, keeping article evidence and intended roles fixed. This tests order sensitivity.

| Route | Earlier bank / 116 | External bank / 80 | External reordered / 80 | External original P95 |
| --- | ---: | ---: | ---: | ---: |
| Raw ridge | 88 | 40 | 41 | 8.33 ms |
| First segment | 96 | 16 | 51 | 8.27 ms |
| Last segment | 18 | 48 | 14 | 8.28 ms |
| Cross-encoder selects segment, then ridge | 96 | 37 | 51 | 42.19 ms |
| 4B selects segment, then ridge | 96 | 51 | 51 | 32.25 ms |
| Direct full glossary | 68 | 43 | 43 | 93.59 ms |

A group is correct only when both the changed and stable author's before-and-after judgments match. On the external wording, the 4B selector fixes 12 groups and loses one versus raw ridge (+13.75 points, descriptive publication-bootstrap 95% interval +6.25 to +21.62). Unsupported promotions fall from 16 to nine. After reordering, all 80 groups produce exactly the same four labels with the 4B selector. The cross-encoder's output changes on 17 groups. Fixed segment choices remain much cheaper and depend strongly on the wording order.

Compared with the direct 4B glossary, selection followed by ridge saves 61.34 ms at external P95 and returns eight more complete groups (+10.00 points, descriptive 95% interval −2.70 to +21.79). Compared with raw ridge, it adds about 24 ms at P95. The external control wording stays at 80/80 with ridge-based methods and without invoking the selector. Timing covers parsing, selection, encoding and predicted roles for a batch containing both edit versions. The phrase banks are authored diagnostics, and the reorder is a targeted perturbation of the same content.

## Reproduce

Run [the inference comparison](../../experiments/credit/role_clause_selection.py) with `FIXTURE AUDIT_JSON BOUNDARY_PHRASES_JSON NISO_EXAMPLES_JSON ROLE_TRAINING_JSON EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT READER_CHECKPOINT NEW_RESULT_JSON` on a Slurm CUDA compute node. For the separate publisher capture, add `--comparison-phrases benchmarks/plos-credit-transfer-v1/phrases.json`. For the earlier test, use `benchmarks/plos-credit-transfer-v1` as the fixture and the audit and phrases from `benchmarks/plos-credit-role-transfer-hard-v1`.
