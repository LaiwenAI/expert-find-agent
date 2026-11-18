# Personal attribution of numerical evidence

Six synthetic scenarios from the [mixed-passage fixture](../bilingual-numeric-passages-v1/README.md) are paired with explicit role statements. The candidate can perform, jointly perform, or only review a test. Another person can supply a seemingly relevant result. Both sentence-block orders are evaluated in English and Chinese. The 192 views cover wrong-person approval, wrong-person conflict, missing personal measurements, and genuine personally attributable conflict.

The direct model reads the requirement and both excerpts. The scope-only control selects comparable measurements without conditioning on the person. The joint check asks the reader to verify comparability and personal execution in one decision per excerpt. The factored method verifies measurement scope and independently checks personal execution only for excerpts passing scope. The two factored routes compare the original numeric values exactly and abstain on conflicting comparable evidence.

| Route | English correct / 96 | Chinese correct / 96 | English false approvals | Chinese false approvals | Chinese P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct reader | 37 | 43 | 17 | 32 | 16.46 ms |
| Scope only | 42 | 42 | 30 | 30 | 29.27 ms |
| Joint check per excerpt | 60 | 76 | 30 | 14 | 29.42 ms |
| Separate scope and personal execution | 84 | 84 | 12 | 12 | 51.45 ms |
| Provided source labels | 96 | 96 | 0 | 0 | — |

Separate checks preserve all 36 valid approvals per language and retain no other person's measurement. The source-only control selects 48 other-person excerpts per language. Every remaining false approval from the factored method uses a personally credited excerpt that lacks a comparable measured result. It succeeds in both excerpt orders for 42/48 groups in each language.

Job `40624` uses frozen Qwen3-4B-Instruct-2507 on RTX 5090. P95 includes prompt preparation, sequential decisions, and exact comparison. The fixture supplies distinct, explicit actors and controlled scope statements. Coauthorship, implicit roles, or attribution in enterprise documents need independent judgments. [Aggregate results](results.json)

Run [the replay](../../experiments/transfer/numeric_evidence_attribution.py) with `SCENARIOS CHECKPOINT NEW_RESULT_JSON`, using `benchmarks/bilingual-numeric-passages-v1/scenarios.json` as the scenario input.
