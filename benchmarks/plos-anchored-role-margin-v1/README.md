# Classify the requested duty before attaching a person-specific clause

The [clause rescue experiment](../plos-anchored-clause-rescue-v1) found that author binding leaves a second error: a clause can name the correct author and describe the wrong duty. This experiment compares scoring candidate clauses independently against assigning the requested activity to one global role before attaching the original clause. Both methods apply the same structural author-scope check.

Reuse the same publisher-note cohorts, two candidate authors, and known initial evidence. The second requested duty is supported for only one candidate. The scoped cross-encoder reuses the earlier top-two clause verifier. The embedding route encodes the request once and compares it with the five canonical activity headings. It selects the best role without seeing the two candidate authors. An indexed clause is eligible for attachment only if it explicitly names the candidate and carries that selected publisher role. Compare the absolute best cosine with its gap over the second-best role as confidence scores. All embeddings and both readers are frozen.

Use the earlier 30 calibration notes and freeze the 10% paper-level error constraint and all thresholds. Evaluate the original held-out wording and an additional hand-written paraphrase per role on the transfer panel. A separate stress test replaces each request with one of five administrative activities outside the supported role inventory. The test notes and author scopes stay the same; these unsupported requests require abstention. Neither the paraphrase test nor the unsupported requests enter calibration. The phrase templates are reused across papers, so the experiment tests transfer across notes and authors within a small vocabulary, with a limited wording check.

Timing includes live request encoding or cross-encoding, selection and binding. It excludes source discovery, acquisition of the initial anchor, catalog construction and service overhead. A valid completion requires both the exact credited role and actual author membership. A clean note has a valid attachment and no invalid one. Paired bootstrap intervals resample author-name groups on the transfer panel and paper IDs on the earlier test. These publisher statements do not independently verify capabilities absent from the indexed note.

## Results

Jobs 42853 and 42855 on ws5 reproduce every decision count and calibrated threshold. Timing below uses the RTX 5090 run 42855. Each outcome cell is valid completions / papers with an invalid attachment.

| Panel | Scoped cross-encoder | Absolute cosine | Top-two cosine gap |
| --- | ---: | ---: | ---: |
| Earlier test (20) | 11 / 1 | 6 / 0 | 6 / 0 |
| Transfer, original wording (45) | 22 / 3 | 12 / 0 | 12 / 0 |
| Transfer, new paraphrase (45) | 1 / 0 | 6 / 0 | 12 / 0 |
| Transfer, unsupported activities (45) | 0 / 0 | 0 / 0 | 0 / 10 |

On the original transfer wording, the embedding ranks the intended role first for all 45 notes before applying the threshold. On the new wording, the same ranker matches 34/45 before thresholding, and the gap route retains 12 clean notes versus one for the scoped cross-encoder. The author-cluster paired interval for the clean difference is +11.1 to +38.6 points. Repeated wording templates limit broader inference. The embedding routes have component P95 near 7.3 ms versus about 9.0 ms for the scoped reranker.

The gap measures separation between the known roles. It accepts unsupported administrative activities on 10/45 notes despite having no valid attachment, whereas the absolute cosine and scoped cross-encoder abstain on these controls. Keep an explicit applicability check, separate from resolving among the known roles. A larger test needs independent wording and organic out-of-inventory requests before any claim about abstention transfers.

[Aggregate results](results.json)

## Reproduce

Restore the pinned publisher snapshots and checkpoints used by the [anchored clause benchmark](../plos-anchored-clause-rescue-v1) and run on a Slurm CUDA node.

```sh
PYTHONPATH=src:. python -m experiments.credit.anchored_role_margin \
  OLD_PUBLISHER_SNAPSHOT TRANSFER_PUBLISHER_SNAPSHOT EMBEDDING_CHECKPOINT \
  RERANKER_CHECKPOINT NEW_RESULTS.json
```
