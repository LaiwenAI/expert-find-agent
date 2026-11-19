# Compile author-bound contribution clauses before queries

Reuse the [real publisher note comparison](../plos-legacy-contribution-text-v1/README.md), with its 30-paper calibration split, 20-paper energy test split, and separate wording for each. Candidate source association and author initials remain supplied, and the program verifies both panel lineage and the previous semantic-verifier outcomes.

At indexing time, accept clauses of the form action text, colon, and a list of author initials. Store each clause's normalized-note span and owners. The compiler has no role-heading dictionary and does not use the adjudicated role pairs. Omit clauses outside this grammar. Encode each distinct action head once with the open Qwen3-Embedding-0.6B and also materialize character 3–5 gram counts.

At reception of a candidate source, use two task descriptions. The no-model route scores query character grams against each materialized action head. The two semantic routes batch the task embeddings into one forward and score the cached vectors. Compare the strongest similarity among that author's clauses with or without subtraction of the strongest score from a clause that omits the author. In each route the weaker of the two request conditions determines acceptance. The runner checks that every selected span resolves to the named author's original statement. Use the same 10% calibration risk protocol for every arm, with strict inequality at the frozen threshold.

The paired timing includes task encoding or character scoring and candidate-level comparisons. It excludes index construction, trace-check assertions and storage. The catalog construction overhead and vector bytes are reported separately. This is a first-query verification component on a supplied note, not a search or free-prose attribution measurement. The earlier panels have been examined before this design.

## RTX 5090 results

Slurm job 42624 reproduces all four earlier verifier thresholds and verdict counts. Building the catalog for 50 papers takes 60.24 ms after the encoder is warm. It stores 247 bound clauses from 28 distinct action heads. Per-note source vectors take 1,011,712 bytes as implemented.

| Route | Correctly ordered pairs / 20 | Accepted supported authors | Accepted wrong authors | Both correct | Paired P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Read scoped sentences with the 0.6B reranker | 17 | 0 | 0 | 0 | 8.76 ms |
| Read full note with the 4B reader | 19 | 15 | 3 | 13 | 23.71 ms |
| Indexed character margin | 18 | 17 | 10 | 8 | 0.16 ms |
| Indexed absolute semantic score | 14 | 0 | 0 | 0 | 8.50 ms |
| Indexed owner-contrast semantic margin | 19 | 16 | 6 | 11 | 8.43 ms |

The semantic contrast restores usable coverage at the frozen threshold while reducing paired P95 by 64.4% relative to the 4B route. It also doubles observed wrong-owner acceptances from three to six. Correct ordering ties the 4B reader at 19/20, with different errors; the paired ordering interval spans −15 to +15 percentage points. Absolute similarity alone accepts no candidate at its frozen cutoff. Character scoring is fast enough for candidate generation, but half of the unsupported authors pass its transferred cutoff.

The comparison supports moving source-side extraction ahead of requests where the contribution-note syntax can be verified. It leaves a calibration problem at similar coverage, even before cross-source matching. Keep publication syntax, source-version validity and ownership explicit in the index; carry opaque text into fallback. The current data do not establish equal-risk operation or unrestricted prose extraction. [Aggregate results](results.json)

## Reproduce

Restore the original XML snapshot using [the pinned capture receipt](../plos-legacy-contribution-text-v1/capture.json). Run with the three existing frozen checkpoints in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.compile_legacy_notes \
    SNAPSHOT QWEN_EMBEDDING QWEN_RERANKER QWEN4B ORIGINAL_RESULTS NEW_RESULTS
```
