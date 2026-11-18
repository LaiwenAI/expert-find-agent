# Bounded verifier adjustments to dense reviewer ranking

Start from the same 32 reviewer-publication pairs as the [live prefix benchmark](../cof-live-prefix-stop/README.md). Rescale the dense score by the top-to-tail range within the request. The final score is the normalized dense score plus `weight * tanh(verifier_margin / 4)`, where the margin is the Qwen3-Reranker yes logit minus its no logit. The final list always has five reviewers in adjusted score order, with dense order breaking ties.

Each unchecked candidate's final score is bounded by its dense score plus or minus the weight. Check fixed eight-candidate blocks in dense order. Stop when the highest remaining upper bound is strictly below the fifth-highest lower bound. Any unresolved candidate would then rank outside the top five. Processing equality conservatively preserves the tie convention. The reference evaluates all blocks using the same prompts, block composition and batch size, preserving the verifier's numerical execution path. The certificate concerns the declared score and ordered list. Reviewer relevance and source support are assessed separately.

## Live results

Job `40466` on an RTX 5090 compares the live routes on the same 73 previously explored CoF SIGIR queries. Weight 0.10 is an exploratory setting considered alongside the six-weight replay below. Timing includes query encoding, dense retrieval, verification and result assembly, with resident models and index.

| Route | Mean checks | P50 / P95 request | Hard P@5 | Soft P@5 |
| --- | ---: | ---: | ---: |
| Dense only | 0.00 | 7.44 / 7.61 ms | 20.00% | 42.19% |
| Exhaustive adjustment | 32.00 | 185.64 / 221.20 ms | 20.00% | 41.92% |
| Bounded adjustment | 12.38 | 62.57 / 121.26 ms | 20.00% | 41.92% |

Bounded evaluation matches the exhaustive ordered expert-source pairs on all 73 requests, cuts checks by 61%, and lowers P95 by 45%. Direct dense retrieval remains faster at the same hard P@5. The hard P@5 difference from dense is zero, with a descriptive paired 95% interval of −1.10 to +1.10 percentage points.

| Weight | Mean checks | Hard P@5 | Soft P@5 |
| --- | ---: | ---: | ---: |
| .025 | 8.88 | 20.27% | 42.47% |
| .05 | 10.52 | 20.00% | 42.19% |
| .10 | 12.38 | 20.00% | 41.92% |
| .20 | 18.74 | 19.45% | 41.92% |
| .40 | 24.00 | 20.00% | 42.74% |
| .80 | 28.82 | 19.18% | 42.19% |

The trace results are exploratory. They support the certificate as an execution control for a bounded scoring rule. The quality contribution requires a verifier or evidence state that raises utility over dense retrieval on a fresh cohort. CoF reviewer labels do not judge the cited publications individually. The bound uses the general stopping principle illustrated by [top-K threshold algorithms](https://research.ibm.com/publications/optimal-aggregation-algorithms-for-middleware).

## Reproduce

Follow the [prefix benchmark setup](../cof-live-prefix-stop/README.md#reproduce) to prepare the CoF SIGIR vectors and input manifest. Run [the live adjustment benchmark](../../experiments/retrieval/cof_bounded_adjustment.py) with `INPUT_MANIFEST EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULT_JSON` on a Slurm CUDA compute node. The six-weight frontier reuses the full-block traces and records check counts without reusing them as latency.
