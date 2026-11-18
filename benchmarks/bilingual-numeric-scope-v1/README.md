# Bilingual scope and quantity counterfactuals

Eight synthetic technical scenarios generate seven matched evidence views in English and Chinese. A view can meet the limit, sit on the boundary, violate it, express the same passing value in another unit, state a planned target, measure the wrong quantity, or use another operating setting. Each language has 20 eligible and 36 ineligible or ungrounded views. The fixtures contain no enterprise requests or expert profiles.

The direct method asks the model whether the evidence establishes the condition. The factored method hides the numerical limit from the scope model. If the scope model identifies a comparable empirical measurement, the precise decimal comparator applies the bound and converts compatible units. The oracle uses construction-time scope labels and checks the comparison in isolation.

| Method | English true / false approvals | Chinese true / false approvals |
| --- | ---: | ---: |
| Qwen3-Reranker-0.6B, direct at 0.90 | 20 / 24 | 20 / 30 |
| Qwen3-Reranker-0.6B, factored at 0.90 | 20 / 24 | 20 / 24 |
| Qwen3-4B-Instruct-2507, direct | 16 / 4 | 19 / 15 |
| Qwen3-4B-Instruct-2507, factored | 20 / 1 | 20 / 2 |
| Provided scope and exact comparison | 20 / 0 | 20 / 0 |

The 0.6B model still approves all 24 mismatched-scope controls in each language when its scope decision licenses deterministic arithmetic. The 4B scope model with exact comparison reduces false approvals and preserves the 20 valid approvals in each language. It misses one wrong-setting control in English and one wrong-setting plus one wrong-metric control in Chinese. Three-way decision accuracy increases from 38/56 to 55/56 in English and from 35/56 to 54/56 in Chinese. The construction oracle verifies all numerical decisions. Measured results apply to these templates. Generalization to paraphrased evidence, multiple measurements in a passage, and real enterprise sources requires validation.

Run [the benchmark](../../experiments/transfer/bilingual_numeric_scope.py) with `SCENARIOS CHECKPOINT NEW_RESULT_JSON`. The default uses the frozen 0.6B reranker locally with MLX. Add `--scope-reader` with the frozen 4B checkpoint on Slurm CUDA to reproduce job `40618`. [Aggregate results](results.json)
