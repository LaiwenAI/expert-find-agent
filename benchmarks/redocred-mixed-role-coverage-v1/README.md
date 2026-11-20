# Test mixed-role proof retrieval at matched recommendation coverage

Test the [learned proof-pair selector](../redocred-learned-proof-pairs-v1)
on two different documentary roles for the same candidate. A request
can ask who wrote one work and directed a different work. The two
people, source and four-sentence budget are supplied. Each method
presents exactly two sentences per atomic condition and uses all four
sentences across the four frozen Qwen3-4B prompts.

## Training and held-out requests

Train the same linear scorer on annotated individual relations from
more sources. An eligible relation names a person and a nonperson
entity, with one or two annotated evidence sentences. Include a
different named person and a different work or other nonperson entity in
the training feature context. The additional names are only context
for comparison features. Their relations are never labeled negative.
The training preferences concern incomplete windows for the one
annotated anchor relation. Exclude all official development and test
titles, the preceding title-disjoint holdout, and the new mixed-role
holdout titles before fitting. No Qwen parameters are updated.

Construct a new mixed-role development and test panel from the
official releases. Use two distinct works and roles, with documentary
support for each role somewhere in the two-person candidate set.
Retain at most one case per source for each of the two strata:
a fully labeled conjunction exists, or neither candidate has both
labels. Deterministic IDs break ties. The additional mixed-role
holdout uses upstream training documents excluded from the fitting
set, with at most 100 cases per stratum. Those titles are disjoint
from all previous same-role evaluation panels. Retain the earlier
same-role panels as a cross-check.

## Controls and outcomes

Use the four existing selectors. For mixed-role queries, the global
name control shares its role cue weight equally across the two role
cues. The other selectors score each condition using its own role.
The learned sum and conjunction selectors use the same model and
proposal set. The conjunction selector prioritizes the weaker
condition for each candidate.

Among two candidates, rank by the weaker verifier margin across
the two conditions. Ties use a deterministic hash of the request key without using the
labels. The top margin is also
the issuance confidence. Evaluate two variants at 35%, 50%, 65% and
100% target coverage:

- **Locked development cutoff.** Freeze one cutoff per selector and
  query family on the official development panel at the target
  coverage. Report the realized test coverage and count of known
  and unjudged matches.
- **Matched test coverage.** Select the same number of top-confidence
  requests per selector on the evaluation set without reading their
  labels. This retrospective ranking diagnostic measures selectivity
  at exactly equal issuance. It is unavailable as a per-query cutoff
  without access to the test score distribution.

For both variants, exactly one candidate appears on issued requests.
Count known full conjunctions among the issued candidates, and count
candidates whose displayed excerpts contain both evidence annotations.
Unjudged conjunctions remain unresolved in the [Re-DocRED
release](https://aclanthology.org/2022.emnlp-main.580/).
Intervals resample titles and describe each tested point without
multiplicity correction. The local wall time includes evidence-window
selection and live verification, outside queues and source discovery.

## Results

Both tasks of Slurm array `44366` ran on an RTX 5090 on ws5 with
rotated arm order and reproduced every non-timing aggregate. The
broader scorer uses 642 annotated claims from 284 titles. The
new mixed-role holdout contains 113 requests from 104 titles.
Only 13 of the requests have any fully labeled match. The official
mixed-role test adds 53 requests from 44 titles, with 12 requests
having a fully labeled match. The table shows known matches among
issued requests, with issuance exactly matched in each row.

| Panel and coverage | Issued | Global names | Lexical pair | Learned sum | Learned conjunction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-role holdout, 50% | 106 | 74 | 64 | 79 | 78 |
| Same-role holdout, 100% | 213 | 93 | 97 | 103 | 102 |
| New mixed-role holdout, 50% | 56 | 6 | 5 | 8 | 8 |
| New mixed-role holdout, 65% | 73 | 7 | 5 | 10 | 10 |
| Official mixed-role test, 50% | 26 | 6 | 8 | 5 | 5 |
| Official mixed-role test, 100% | 53 | 10 | 9 | 9 | 9 |

The learned sum gains five known matches and six fully annotated
citations at 50% matched coverage in the same-role holdout.
The paired change in known matches per request is +2.35 points
with a 95% interval of −1.87 to +6.60 points. At 100% coverage
its paired interval is +0.93 to +8.52 points. In the new mixed-role
panel at 50% coverage, both learned routes add two known matches
and one annotated citation. The paired interval has a lower bound
of zero. The official mixed-role test fails to reproduce the gain.

Freezing the 50% development cutoff changes actual issuance in the
new mixed-role panel. The global baseline issues 28
recommendations, learned sum 29 and learned conjunction 33.
All three give six known matches. The matched-coverage diagnostic
therefore provides only limited evidence of a ranking improvement,
and the development cutoff provides no improvement at this point.
The sum objective at least ties the weakest-condition objective
at 50% on all four test groups. The evidence favors retaining a
learned sum control and gives no reason to make the weakest-claim
objective the default. On the new mixed-role holdout, the learned
sum runs at 54.40–54.44 ms P95, versus 51.78–51.90 ms for
global names.

[Complete aggregate results for both execution orders](results.json)

## Closest controller comparison

[S2G-RAG (ACL 2026)](https://aclanthology.org/2026.acl-long.1185/)
already predicts structured evidence gaps and turns them into search
queries. A first-query expert search contribution would need to
isolate what a candidate-specific, owner-bound state buys against a
generic gap judge at the same source, token and inference budgets.
The supplied-document experiment here tests one part of that question:
whether selecting evidence jointly across each person's conditions
improves their ranking and issuance.

## Reproduce

Use the checkpoints and public data from the proof-pair benchmark.
Run in a Slurm GPU allocation with a fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.mixed_role_window_coverage \
  REDOCRED_DIR QWEN4B NEW_RESULTS --order-offset 0
```

Repeat with order offset 1.
