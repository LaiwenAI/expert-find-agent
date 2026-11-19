# Verify owner-linked roles with the retrieval embedding

Use the same publication components, request phrases, candidate records, and schedule budgets as the [live MiniLM witness experiment](../plos-calibrated-witness-search-v1/README.md). Split 60 calibration and 118 test requests by publisher components. The online verifier sees task positions, source slots and candidate declarations. The intended role labels enter calibration and scoring of outcomes.

For each request, obtain its query vectors in one Qwen3-Embedding-0.6B forward. Reuse those vectors for two heads. The first computes similarity to the 14 NISO role definitions. The second uses ridge regression on 84 previously authored task descriptions, embedded with the same query prompt as retrieval. Verify no exact normalized overlap between the training phrases and the public evaluation phrases. No model parameters are updated.

For each candidate declaration that the schedule opens, compare the highest task score over the author's published roles against the highest over all other roles. Require the former to exceed the latter by the calibrated margin. This forces the evidence to support every role in the corresponding score band. Zero is the lowest admissible cutoff. Missing declarations provide no support. Return an author only if all requested tasks pass under the same owner and the required source slots. Retain source round robin, owner mean, and adaptive bottleneck with budgets of two and four reads.

Choose the three cutoffs by the same 10%, 20%, and 30% calibration error caps as the MiniLM reference, using the same deduplicated publisher-role checks. Report false-role decisions separately from unsupported author recommendations. The comparison to MiniLM uses matched calibration targets and a prior job. Test false acceptance need not match. Timing includes query encoding, local scoring, scheduling and in-memory role membership checks. It excludes network, external document access, parsing and offline bank construction. The source roles are structured, source scope is supplied, and publications and phrasings recur within the test split.

## RTX 5090 results

Job 42384 uses positional task IDs in the live verifier and reproduces the decisions from job 42379. Both heads meet the 10% calibration target at the zero cutoff, so all three policy caps return the same field-check decisions. On the 35 wrong-role and 86 support calibration checks, definitions accept 2 and 79, ridge accepts 3 and 80, and the earlier MiniLM run accepts 3 and 24 at its 10% gate.

| Two-read route at 10% calibration target | Positive requests recovered / 59 | Unsupported returns on negative requests / 59 | Both variants exact / 59 | Component P95 |
| --- | ---: | ---: | ---: | ---: |
| MiniLM, owner mean, batched (prior run) | 0 | 0 | 0 | 10.57 ms |
| Definitions, source round robin | 28 | 1 | 27 | 8.59 ms |
| Definitions, owner mean | 34 | 1 | 33 | 8.60 ms |
| Definitions, adaptive bottleneck | 30 | 1 | 29 | 8.58 ms |
| Ridge, owner mean | 31 | 1 | 30 | 8.58 ms |

Owner-mean routing with definition scores finds six additional positives over source round robin using the same 236 reads and with one unsupported request in both arms. Four reads recover 35 positives with one unsupported request for every definition-based schedule. The adaptive schedule uses 215 reads versus 276 for each fixed schedule. Component P95 stays near 8.6 ms, with the shared encoder on the critical path. The earlier exact-field replay with supplied role labels recovered 58/59 positive requests in two reads with owner mean. This difference bounds the remaining task-interpretation opportunity on this fixture. Definition Top-1 maps 31/40 unique test phrases correctly and 7/10 phrases absent from calibration. Ridge maps 30/40 and 6/10. These descriptive comparisons reuse phrases and a small set of source components.

The recommendation improvement is measured at the same calibration target, with one additional unsupported test return relative to MiniLM. It depends on available publisher declarations and a closed role inventory. The rerun does not test free-text role extraction, new publishers, or end-to-end enterprise search. [Aggregate results](results.json)

## Reproduce

Use the same pinned publisher projection, glossary and Qwen embedding checkpoint as the [live witness schedule study](../plos-calibrated-witness-search-v1/README.md). Supply its aggregate output and the [original synthetic role training phrases](../plos-credit-edits-v1/role-training.json). Run in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.shared_embedding_witness_gate \
    PUBLISHER_PROJECTION NISO_EXAMPLES ROLE_TRAINING QWEN_EMBEDDING PRIOR_RESULTS NEW_RESULTS
```
