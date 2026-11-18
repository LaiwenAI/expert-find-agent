# Live prefix stopping for contributor attribution

Reuse the eight test articles from the [PLOS CRediT requirement-edit fixture](../plos-credit-edits-v1/README.md). Deduplicate the edits to 131 article-and-role-pair requests and render each role with a pinned [NISO task example](https://zenodo.org/records/18421449). The article is supplied. All author declarations enter the candidate pool in byline order, which acts as a fixed tie break. The model judges only the requested author's statement for each check.

Compare a single full batch, exhaustive batches of four, batches of four with prefix stopping, and stopping with batch width capped by the remaining number of citations needed. Both stopping policies scan the same author order and stop once they obtain K positive decisions. If the observed prefix decisions are unchanged, later candidates cannot change the first K citations. Batch shape can change model decisions, so the replay compares actual citation sequences and common-prefix labels.

## Results

Slurm job `40456` runs frozen Qwen3-4B-Instruct-2507 on an RTX 5090. Each of the 131 requests is measured twice at each K, with rotated arm order. Candidate pools contain two to eight authors. P95 measures prompt construction, tokenization, local model execution, and citation assembly after warm-up. Retrieval and service transport are outside the timing.

| K | Policy | Mean checks | Mean batches | P95 | Citations identical to exhaustive batches of four |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 | Full batch | 5.59 | 1.00 | 69.77 ms | 252 / 262 |
| 3 | Exhaustive batches of four | 5.59 | 1.63 | 76.09 ms | 262 / 262 |
| 3 | Stop, batches of four | 5.34 | 1.55 | 74.63 ms | 262 / 262 |
| 3 | Stop, batch sized to remaining quota | 5.30 | 2.07 | 82.25 ms | 250 / 262 |
| 5 | Full batch | 5.59 | 1.00 | 69.81 ms | 250 / 262 |
| 5 | Exhaustive batches of four | 5.59 | 1.63 | 75.95 ms | 262 / 262 |
| 5 | Stop, batches of four | 5.59 | 1.63 | 75.95 ms | 262 / 262 |
| 5 | Stop, batch sized to remaining quota | 5.56 | 1.71 | 79.72 ms | 260 / 262 |

At K=3 the fixed-shape stopping rule skips checks on 11 of 131 requests, consistently across both repetitions. At K=5 it skips none. Each fixed-shape prefix decision matches its exhaustive reference in both repetitions. At K=3 the one-batch run is 4.85 ms faster at P95, with different citations in ten measurements. It returns the exact gold top K on 33/131 requests, as does the fixed-shape stopping route. Both share the verifier's model errors on these NISO paraphrases. The result favors including a one-batch verifier as a baseline for short candidate lists and preserving the verifier batch schedule when testing exactness. Serving overhead under larger batches or concurrent requests remains unmeasured. [Related serving work](https://doi.org/10.1145/3694715.3695963).

## Reproduce

Run [the live replay](../../experiments/credit/live_certified_attribution.py) with `FIXTURE NISO_EXAMPLES_JSON QWEN3_4B_CHECKPOINT NEW_RESULT_JSON` on a CUDA compute node within a Slurm allocation. The fixture is the requirement-edit directory. Use the same frozen Qwen3-4B-Instruct-2507 checkpoint and published examples as the [attribution benchmark](../plos-credit-edits-v1/README.md#shared-parsing-before-attribution).
