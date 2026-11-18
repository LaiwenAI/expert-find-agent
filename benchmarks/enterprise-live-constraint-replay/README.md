# Enterprise hard-constraint replay with a local verifier

The diagnostic uses eight private development requests selected for explicit hard criteria. Two verbatim clauses per request are supplied to a frozen Qwen3-Reranker-0.6B on Apple M4 with MLX. The projection provides seven fixed candidate experts per request and 209 candidate–source views overall. All visible sources remain available for every clause. Character-bigram overlap orders witnesses and uses source ID as a tiebreaker. The verifier sees the project title, one criterion, and one source. Two of 418 prompts reach the 768-token cap.

A source read supplies both scores for the same candidate and document. The quality threshold determines an operational model decision. It does not establish expertise or individual contribution. All hypothetical schedules use the same scored trace. The result compares distinct source reads, without translating them into wall-clock savings.

| Model threshold | Candidate–criterion slots with any support | Experts with both criteria | Mean grouped reads, fixed / adaptive |
| --- | ---: | ---: | ---: |
| 0.90 | 1/112 | 0/56 | 26.125 / 26.125 |
| 0.50 | 4/112 | 1/56 | 26.000 / 26.000 |

All schedules match exhaustive expert order and citations. Most candidate views contain no source scored above threshold for either criterion. With both criteria unconfirmed, a candidate cannot be eliminated until every alternative source for at least one criterion has been examined. This leaves almost no opportunity for early stopping. The experiment cannot distinguish missing evidence from verifier misses without independent labels.

The optional local packet samples two high-scored and one low-scored source per criterion. Its 48 private views hide scores and selection order. The sampling design targets error inspection. Human labels are pending, and the sealed evaluation partition was unused.

Run [the trace and scheduling replay](../../experiments/retrieval/enterprise_live_constraint_replay.py) with `DEVELOPMENT_JOBS ANCHORED_REQUIREMENTS CHECKPOINT NEW_RESULT_JSON` in the MLX environment. The requirements file maps development query IDs to arrays of verbatim clauses. Add `--review NEW_PRIVATE_JSONL` for a local review packet. The release includes aggregate results and code. Raw data and the case-level outputs stay local.
