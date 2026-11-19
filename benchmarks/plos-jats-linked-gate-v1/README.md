# Compile publisher declarations and check plausible role interpretations

Use the [178 paired contributor requests](../plos-linked-witness-dispatch-v1/README.md), with the two sources and the scope of each clause supplied. Rebuild the declaration postings from the 17 original publisher JATS files. Verify each snapshot's SHA-256, the original contributor ordinal, the valid ORCID linkage in both directions and the publisher-declared role set before compiling a role-to-owner posting per source. No article-wide role union can satisfy a personal requirement. The published results contain aggregate counts and include no raw JATS or author identities.

Classify the three natural-language phrases in one frozen Qwen3-Embedding-0.6B forward. Compare definition cosine with ridge trained on 84 authored phrases from the earlier controlled role study. Derive two calibrated uncertainty cutoffs from one NISO example per role. Keep the other 35 NISO phrases separate for testing and hash-select them consistently for paired requests. Reject exact normalized overlap across all three phrase sets. These are empirical cutoffs from fourteen examples.

Preserve three distinct requested roles, as in the generation protocol. Within each plausible interpretation, intersect the same contributor's declarations at the required sources. Return an owner only when every plausible interpretation supports that owner. If the interpretation sets disagree, mark the result unresolved, even if a subset of owners is safe to return. An empty consensus is conclusive only when every admitted interpretation has an empty answer. Risk control requires the intended interpretation to be in the admitted set. The live text control reruns two mean-ranked owned reads on exactly the same held phrases and uses the generic frozen Qwen reranker to check the rendered definitions. A no-return from this baseline is counted as an empty answer.

## Results on RTX 5090

| Route | Correct positive / 89 | Unsupported negative / 89 | Unresolved / 178 | Exact paired decisions / 89 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Live text, two owner-aware reads | 15 | 6 | 0 | 12 | 25.49 ms |
| Definition cosine and JATS | 35 | 5 | 25 | 30 | 8.65 ms |
| Ridge top role and JATS | 64 | 7 | 13 | 59 | 8.65 ms |
| Ridge consensus, 90% calibration cutoff | 51 | 1 | 38 | 47 | 8.66 ms |
| Ridge consensus, widest calibration cutoff | 12 | 0 | 111 | 9 | 8.67 ms |

Job 41722 verifies all 17 raw XML hashes and link records in 41.84 ms offline. Online timing includes one phrase-embedding forward and candidate assembly, with XML parsing outside the timer. The text baseline includes its own encoding and live source checks. The 90% cutoff improves positive recovery by 36 and reduces unsupported negative requests by five against the same-cohort text baseline. It also marks 38 requests unresolved. It includes the intended interpretation in 134/178 requests, versus 107 for the ridge point prediction. The widest cutoff includes all intended interpretations here and sharply reduces positive coverage. This is a controlled, reused article panel with a known role taxonomy and supplied source bindings. These figures do not establish a population risk threshold or the applicability of structured verification to arbitrary enterprise evidence. [Aggregates](results.json)

## Reproduce

Rebuild the pinned [source projection](../plos-scoped-search-v1/README.md), place its relevant publisher snapshots in `RAW_DIRECTORY` as `<DOI suffix>.xml`, and use the same 84-phrase [training fixture](../plos-credit-edits-v1/role-training.json). Run with verified open checkpoints on a Slurm CUDA node.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.jats_linked_gate \
  PINNED_PROJECTION NISO_EXAMPLES ROLE_TRAINING RAW_DIRECTORY \
  EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS
```
