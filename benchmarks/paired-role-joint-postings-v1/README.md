# Recover a role-compatible source pair within one verifier forward

The [static source-postings experiment](../paired-role-source-postings-v1)
found five keyword positives whose two sources were already in the top
24 but whose declarations failed the static role band. Keep the
source search and strict agreement route unchanged when a source pair
has been selected or the request has spent its verifier forward.
Otherwise, compare these role predicates before joining postings.

- **Static posting filter.** Require the entire prototype band.
- **Dense singletons.** Choose one label per activity from the combined
  ridge and prototype scores without a verifier forward.
- **Student agreement.** Require the distilled linear student and
  the dense leader to agree for each activity. Use the frozen learner
  and the same PLOS calibration procedure as the
  [lightweight role study](../lightweight-paired-role-v1).
  The student reads the resident activity vectors and no source or
  author identifiers.

All variants use the same top-24 lists, same-owner postings join and
pair order. Verify both source clauses of the top predicted pair in
one reranker forward. Each source must pass the prior frozen cutoff.
Count a verifier dispatch separately from the number of clause-source
rows. Shared matched observations do not discount either route's
measured work.

Replay the same 108 public structured requests with and without
each intended source. Neither the policy nor the student sees eLife
labels. The stage and alternatives were proposed after this panel had
been examined. Two request orders check repeatability. The evaluation
measures source and role evidence under the one-forward cap. Service
latency and independent raw-query results are outside this study.

## Results

Both tasks in Slurm array 46241 completed on ws3 RTX 5090.
The two request orders reproduce all results and emitted proof
signatures. The stock proof matches the archived replay.

| Policy | Keyword strict / adverse | Excerpt strict / adverse | Full-availability source forwards / rows | Role forwards |
| --- | ---: | ---: | ---: | ---: |
| Stock | 12 / 0 | 18 / 0 | 4 / 4 | 22 |
| Static posting filter | 12 / 0 | 18 / 0 | 5 / 6 | 22 |
| Dense singletons | 15 / 0 | 18 / 0 | 8 / 12 | 22 |
| Student agreement | 13 / 0 | 18 / 0 | 6 / 8 | 22 |

Each view contains 27 positive requests and 27 role swaps. Of the
seven positive keyword requests eligible for the new action, the
static filter admits no gold pair. Dense singletons put three gold
pairs first and the student agreement retains one. Both sources
pass confirmation on all three top gold pairs. Excerpt requests
never enter this particular fallback at full source availability.

The 216 one-source-removal cases yield zero issued substitutes for
each policy. Dense singletons require eight extra source forwards
across those removal cases relative to the stock path. The two
wordings and source-removal variants share the underlying papers.
The lack of observed adverse issuances gives limited evidence about
risk on new paper families.

This replay supports testing dense singletons with both source sides
verified as the next candidate. The student reduces model dispatches
and removes two otherwise successful cases. The per-request limit
holds for all policies, while realized work increases. Measure tail
latency under matched arrivals and strict quality on unseen paper
families before claiming a first-query quality-cost improvement.
See [aggregate results](results.json).

## Reproduce

Restore the [prospective inputs](../paired-role-prospective-service-v2#reproduce)
and the [lightweight reference and PLOS inputs](../lightweight-paired-role-v1#reproduce).
Run inside Slurm GPU, repeating with `--reverse` and a fresh output.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.credit.paired_role_prospective_service \
  ELIFE_PROJECTION ELIFE_MANIFEST benchmarks/paired-role-verification-v1/results.json \
  NISO_EXAMPLES ROLE_TRAINING SCORE_BANDS SOURCE_FROZEN ROLE_FROZEN \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_Both tasks in Slurm array 46241 completed on ws3 RTX 5090.
The two request orders reproduce all results and emitted proof
signatures. The stock proof matches the archived replay.

| Policy | Keyword strict / adverse | Excerpt strict / adverse | Full-availability source forwards / rows | Role forwards |
| --- | ---: | ---: | ---: | ---: |
| Stock | 12 / 0 | 18 / 0 | 4 / 4 | 22 |
| Static posting filter | 12 / 0 | 18 / 0 | 5 / 6 | 22 |
| Dense singletons | 15 / 0 | 18 / 0 | 8 / 12 | 22 |
| Student agreement | 13 / 0 | 18 / 0 | 6 / 8 | 22 |

Each view contains 27 positive requests and 27 role swaps. Of the
seven positive keyword requests eligible for the new action, the
static filter admits no gold pair. Dense singletons put three gold
pairs first and the student agreement retains one. Both sources
pass confirmation on all three top gold pairs. Excerpt requests
never enter this particular fallback at full source availability.

The 216 one-source-removal cases yield zero issued substitutes for
each policy. Dense singletons require eight extra source forwards
across those removal cases relative to the stock path. The two
wordings and source-removal variants share the underlying papers.
The lack of observed adverse issuances gives limited evidence about
risk on new paper families.

This replay supports testing dense singletons with both source sides
verified as the next candidate. The student reduces model dispatches
and removes two otherwise successful cases. The per-request limit
holds for all policies, while realized work increases. Measure tail
latency under matched arrivals and strict quality on unseen paper
families before claiming a first-query quality-cost improvement.
See [aggregate results](results.json). \
  --joint-postings-reference CASCADE_RUN --plos PLOS_PROJECTION
```
