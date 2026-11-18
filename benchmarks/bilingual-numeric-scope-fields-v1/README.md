# Typed scope decisions for personally attributed measurements

The replay compares the binary scope prompt from the [personal attribution benchmark](../bilingual-numeric-attribution-v1/README.md), the same prompt with left-padded batching, and a batched prompt that explicitly distinguishes a valid measurement from setting mismatch, subject or metric mismatch, and unmeasured targets. A separate personal-execution decision and exact unit-aware comparison are shared. Batches contain the excerpts from one request.

The 192 development views repeat the earlier six scenarios. The 128 new views use four new subjects, metrics and operating settings, with a paraphrase of the required setting in each positive excerpt. Both cohorts use the same attribution templates and paired excerpt reversals. Neither cohort contains natural enterprise documents.

| Cohort | Language | Route | Correct | False approvals | Request P95 |
| --- | --- | --- | ---: | ---: | ---: |
| Development | English | Sequential binary | 84/96 | 12 | 48.33 ms |
| Development | English | Batched binary | 84/96 | 12 | 39.14 ms |
| Development | English | Typed scope | 86/96 | 10 | 45.88 ms |
| Development | Chinese | Sequential binary | 84/96 | 12 | 51.69 ms |
| Development | Chinese | Batched binary | 84/96 | 12 | 39.28 ms |
| Development | Chinese | Typed scope | 88/96 | 8 | 46.02 ms |
| New scenarios | English | Sequential binary | 60/64 | 4 | 51.93 ms |
| New scenarios | English | Batched binary | 60/64 | 4 | 39.46 ms |
| New scenarios | English | Typed scope | 60/64 | 4 | 46.22 ms |
| New scenarios | Chinese | Sequential binary | 64/64 | 0 | 51.88 ms |
| New scenarios | Chinese | Batched binary | 60/64 | 4 | 39.52 ms |
| New scenarios | Chinese | Typed scope | 64/64 | 0 | 45.97 ms |

All routes retain the 36 valid approvals per development language and 24 per new-scenario language. In the new Chinese views, vanilla batching changes four decisions on two scenario pairs under excerpt reversal. The flipped scope decisions approve a personally executed test under the wrong setting. The typed prompt restores those four verdicts, but still differs from the sequential baseline on some witness choices. New-scenario English gains are limited to latency versus sequential verification.

Job `40627` uses frozen Qwen3-4B-Instruct-2507 on RTX 5090. The three routes rotate execution order by request. P95 covers prompt preparation, verifier calls and comparison for a warm request that already has two excerpts and a scoped constraint. The result motivates measuring both verdict and citation agreement when changing tensor shapes. It provides no estimate for enterprise search latency or for performance on unseen attribution templates. [Aggregate results](results.json)

Run [the replay](../../experiments/transfer/numeric_scope_fields.py) with `OLD_SCENARIOS NEW_SCENARIOS CHECKPOINT NEW_RESULT_JSON`. Use `benchmarks/bilingual-numeric-passages-v1/scenarios.json` for the development set and [scenarios.json](scenarios.json) for the new set.
