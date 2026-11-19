# Owner-aware selection of source-scoped contributor witnesses

Use the pinned [PLOS source projection](../plos-scoped-search-v1/README.md) of 171 publications, with publisher-declared roles and dataset-local keys derived from validated ORCID identifiers. A request supplies two source IDs and three contribution clauses split across the two sources. Gold requires one person to satisfy every clause in the specified source. Candidate evidence units are individual author declarations within an article. Statements are rendered from roles using the [NISO role definitions](https://zenodo.org/records/18421449). Query phrases are drawn from the examples using a stable family hash and use consistent wording across paired variants. The benchmark isolates witness dispatch after source scope and identity are available.

For every pair of articles linked by a declared contributor, generate a positive request. Create a paired negative by swapping the source bindings of two roles or replacing a single role with another role declared in that same article. In both cases all requested roles occur somewhere in the specified articles, while no contributor satisfies the changed conjunction. Freeze at most six swaps and four role edits per article pair by stable hashes. This deliberately selects attribution-sensitive cases and reuses publisher records and phrases.

Every arm encodes the same three phrases in one Qwen3-Embedding-0.6B forward. The indexed declaration text represents its roles without the author or article name. All statement scores are available to the scheduler. Exact declaration roles become available only when that statement is opened. At most two or four statements can be opened. The final check requires an exact publisher-backed witness in each source for the same identity.

| Policy | Opening order |
| --- | --- |
| Independent | Prefer a statement with a high score on any query phrase |
| Source-scoped | Prefer high scores only for phrases attached to the statement's source |
| Owner mean | Prioritize the owner whose two-source plan has the strongest average score |
| Owner bottleneck | Prioritize the owner whose weakest required clause match is strongest |
| Adaptive bottleneck | Recompute after each read. Drop a plan if the opened record fails any required clause on that side; count confirmed clauses before selecting the next witness |

For a failure control, on the same opened statements, union the verified roles at the article level and attribute them to linked authors. This measures the cost of dropping owner specificity under identical reads. Every owner-aware route uses an exact structured verifier in this replay. Its lack of unsupported outputs follows from that constraint and does not estimate errors of a learned verifier.

Warm P95 includes live query encoding, similarity calculations and statement dispatch. The timed read uses a resident lookup after the statement is selected. It excludes source discovery, natural-language scope binding, raw declaration parsing, transport and serving. The fixture uses anonymized contributor keys. It establishes publisher-declared roles on these records, without measuring wider expertise or industrial relevance.

## Exact declaration replay

Job 41693 covers 11 article pairs out of 25 available declared links. It uses 89 matched families, each comprising one positive and one negative, with 45 source swaps and 44 role edits. Results are descriptive across repeated publications and contribution phrases.

| Budget | Policy | Positive exact / 89 | Both variants exact / 89 | Read count / 178 | P95 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | Independent | 19 | 19 | 356 | 8.53 ms |
| 2 | Source-scoped | 39 | 39 | 356 | 8.53 ms |
| 2 | Owner mean | 88 | 88 | 356 | 8.56 ms |
| 2 | Owner bottleneck | 81 | 81 | 356 | 8.55 ms |
| 2 | Adaptive bottleneck | 81 | 81 | 337 | 8.54 ms |
| 4 | Independent | 79 | 79 | 684 | 8.54 ms |
| 4 | Source-scoped | 82 | 82 | 684 | 8.54 ms |
| 4 | Owner mean | 89 | 89 | 436 | 8.55 ms |
| 4 | Owner bottleneck | 89 | 89 | 436 | 8.55 ms |
| 4 | Adaptive bottleneck | 89 | 89 | 378 | 8.55 ms |

With two reads, averaging the three scoped matches per owner finds 88 of 89 exact positive results. The more conservative bottleneck score finds 81. With four available reads up to 89 exact positives are recoverable, and adaptation uses 378 reads versus 436 for the static owner policy. All scope-aware final checks reject all 89 paired negatives. Removing owner specificity with the same four-read source-scoped schedule misattributes roles in 74 negative cases. The identical timing scale across routes reflects the common query encoding, which dominates the resident lookup replay.

## Live text verification

Job 41702 reruns the same 178 requests with Qwen3-Reranker-0.6B. After each selected text read, score each clause attached to the article. Use predicted role checks to update the search state and issue an answer. The declared role sets only determine the benchmark labels. The prompts contain the rendered definition text and the clause phrase. Identity and scope remain metadata. Compare a fixed margin threshold of zero with a stricter threshold of one. These cutoffs are exploratory, without risk calibration.

| Policy | Positive requests with a correct return / 89 | Negative requests with an unsupported return / 89 | Both variants exact / 89 | P95 |
| --- | ---: | ---: | ---: | ---: |
| Source-scoped, four | 16 | 6 | 11 | 42.42 ms |
| Owner mean, two | 19 | 8 | 13 | 24.94 ms |
| Adaptive, four | 19 | 9 | 12 | 33.19 ms |
| Adaptive, four, stricter | 2 | 2 | 2 | 32.94 ms |

The two-read owner-mean route saves time relative to four-read source-scoped verification, but accepts eight negative requests and recovers just 19 positives. Its exact-reader version recovers 88. The stricter adaptive variant reduces unsupported returns from nine to two and loses 17 of 19 correct positive returns. Identity-aware scheduling avoids mixing owners. Semantic verdicts can still attach an unsupported claim to the correct record. The formatter uses declarations derived from publisher roles, so evaluation of raw publisher prose remains open. [Exact replay aggregates](replay.json) · [Live verification aggregates](live.json)

## Reproduce

Rebuild the pinned source projection and glossary using the [previous source experiment](../plos-scoped-search-v1/README.md). Use the verified open embedding checkpoint and a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.linked_witness_dispatch \
  PINNED_SOURCE_PROJECTION NISO_EXAMPLES EMBEDDING_CHECKPOINT NEW_RESULTS

PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.live_linked_witness_verification \
  PINNED_SOURCE_PROJECTION NISO_EXAMPLES EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_LIVE_RESULTS
```
