# Diagnostic expansion of enterprise evidence sources and windows

Use the 25 multi-criterion requests from the same frozen 80-request development cohort as the [gap verification benchmark](../enterprise-gap-verification-v1/README.md). Keep its query parser, exact dense retrieval, first five experts, ownership checks, two-wave opening reference and frozen verifier. Compute deeper source ranks from the same full-source score function and assert that the previous source prefix remains intact.

Start with the first 800 characters of the top two sources for each expert. For still unsupported criteria, expand to the opening spans of sources ranked three through eight, or add a later window in the two original sources. The later window has at most 800 characters and is selected by Chinese bigram and English token overlap. Its possible starting offsets include a 400-character grid and the last full window. The fourth arm allows later windows in the deeper sources as well. Expanded pairs are scored separately and may add positive evidence. Every arm retains previously supported conditions and the original citation when available. The test does not score conditions beyond the original length limit. It rejects batches containing truncated comparisons.

| Sources and spans | Added criterion–excerpt scores | Supported expert–criterion slots / 365 | Experts with all criteria supported | Requests with at least one such expert |
| --- | ---: | ---: | ---: | ---: |
| First two openings | 0 | 46 | 3 | 2 |
| First eight openings | 1,839 | 48 | 3 | 2 |
| First two, plus later spans | 212 | 52 | 3 | 2 |
| First eight, plus later spans | 2,934 | 56 | 3 | 2 |

The 2,934 extra scores produce ten more model-supported slots and no additional complete recommendation. This comparison leaves the expert shortlist fixed. It narrows the benefit of increasing read depth under this model and protocol. It does not distinguish missed support from missing evidence, or assess alternative candidate retrieval. Scores describe the model, not independent judgments. No latency gain is inferred from these diagnostic comparisons. The sealed cohort was not used. [Aggregate results](results.json)

## Local review

The private packet contains 19 displayed excerpts, including seven newly supported by the verifier and twelve from criteria with no new support. Each group takes at most one case per request. Within a request, select a high-overlap excerpt, then use a hash order to select across requests. The cases intentionally enrich disagreements with the shallow view and challenging negatives. They do not estimate population rates.

Two reviewers can independently annotate whether the displayed source supports the criterion. Keep the case-to-model mapping with the coordinator. Judge support in the excerpt, and treat personal contribution and ownership as separate questions. The [scoring script](../../experiments/retrieval/score_evidence_extent_review.py) checks the frozen input hashes, reviewer IDs and immutable text. Its bounds retain uncertainty, missing reviews and unresolved disagreements. Prepare the private packet manifest with the case count and SHA-256 hashes of both JSONL files under `sha256` in `packet.json` before sending the blinded copy for review. No human judgments have been collected for this packet.

## Reproduce

Use the authorized development projections and checkpoints listed in the linked benchmark. Job 41146 ran on one RTX 5090.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_evidence_extent \
  --queries DEVELOPMENT_QUERIES --corpus DEVELOPMENT_SOURCES --models CHECKPOINT_DIRECTORY \
  --output NEW_AGGREGATE --review-dir PRIVATE_REVIEW_DIRECTORY
```

The release contains aggregates and code. The private review inputs are not distributed.
