# Diagnose joint source-recovery errors and compare resident role gates

Keep the earlier 27-pair development panel and the 12-pair
[held eLife panel](../elife-joint-postings-service-v1). Reproduce each
archived service response for the cascade and the dense joint fallback.
Local replays run the same algorithms and compare logical verifier work.
Source, role and author hints are unchanged from the service experiment.

Test two gates before calling the source reranker on a fallback.
The student gate requires both combined dense leaders to agree with
the trained linear student and its frozen PLOS margin. The two-bank
gate requires agreement among the existing ridge and prototype
leaders for both role clauses. All representations are resident.
Neither gate adds a neural forward and both retain the one-forward
source or role cap. An admitted fallback verifies both sides with the
same reranker batch as before.

For all outputs added by the dense fallback, separate wrong source,
wrong role and wrong contributor, and count which correct or adverse
outputs each gate suppresses. A role-swap control can have no correct
contributor even when the selected sources are correct. Classify its
error from the conjunction of role and person. Test both request orders
on both panels. This is a diagnostic on already observed panels.
Quality and service latency of any revised gate require new data.

## Results

Both request orders for each panel complete on ws3 RTX 5090
in Slurm array 46267. All counts and output signatures match across
orders. The archived service proof signatures also match.

| Panel | Cascade keyword strict / adverse | Dense fallback strict / adverse | Student guard strict / adverse | Two-bank guard strict / adverse |
| --- | ---: | ---: | ---: | ---: |
| Development, 27 keyword positives | 12 / 0 | 15 / 0 | 13 / 0 | 13 / 0 |
| Held, 12 keyword positives | 10 / 0 | 11 / 1 | 10 / 0 | 10 / 0 |

The held adverse result is a swapped-role negative control. The
sources are correct, at least one role label is wrong, and the
returned contributor fails the requested combination. Neither panel
contains a positive and corresponding swap that both issued an
identical proof. The observed error concerns role interpretation
even with confirmed sources and a declared contributor.

On held keywords, both gates skip both fallback forwards, removing
one wrong result and one strict proof. They retain the cascade's
10 strict proofs. On development keywords, the gates skip two of
the four fallback forwards and preserve one of three extra strict
proofs. The two gates coincide on these cases. Every excerpt result
is unchanged. Neither gate separates the two held fallback outcomes.
This comparison motivates a direct role check within the same
source-and-role budget. See [aggregate results](results.json).

## Reproduce

Restore the two panel projections and frozen inputs from the
[service study](../elife-joint-postings-service-v1#reproduce).
Use a single archived service run as the response reference for its
matching projection. Execute inside a Slurm GPU allocation and use a
fresh output for each order and panel.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.credit.paired_role_prospective_service \
  ELIFE_PROJECTION ELIFE_MANIFEST benchmarks/paired-role-verification-v1/results.json \
  NISO_EXAMPLES ROLE_TRAINING SCORE_BANDS SOURCE_FROZEN ROLE_FROZEN \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_Both request orders for each panel complete on ws3 RTX 5090
in Slurm array 46267. All counts and output signatures match across
orders. The archived service proof signatures also match.

| Panel | Cascade keyword strict / adverse | Dense fallback strict / adverse | Student guard strict / adverse | Two-bank guard strict / adverse |
| --- | ---: | ---: | ---: | ---: |
| Development, 27 keyword positives | 12 / 0 | 15 / 0 | 13 / 0 | 13 / 0 |
| Held, 12 keyword positives | 10 / 0 | 11 / 1 | 10 / 0 | 10 / 0 |

The held adverse result is a swapped-role negative control. The
sources are correct, at least one role label is wrong, and the
returned contributor fails the requested combination. Neither panel
contains a positive and corresponding swap that both issued an
identical proof. The observed error concerns role interpretation
even with confirmed sources and a declared contributor.

On held keywords, both gates skip both fallback forwards, removing
one wrong result and one strict proof. They retain the cascade's
10 strict proofs. On development keywords, the gates skip two of
the four fallback forwards and preserve one of three extra strict
proofs. The two gates coincide on these cases. Every excerpt result
is unchanged. Neither gate separates the two held fallback outcomes.
This comparison motivates a direct role check within the same
source-and-role budget. See [aggregate results](results.json). \
  --joint-guard-reference MATCHED_SERVICE_RUN --plos PLOS_PROJECTION
```
