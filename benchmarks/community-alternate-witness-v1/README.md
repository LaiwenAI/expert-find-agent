# Spend five parallel reads on new authors or alternate answers

The [Unix & Linux budget study](../community-read-budget-v1) found
50 requests without a known answer among the first ten account
representatives. Separate whether the account is outside the shortlist
or the chosen answer differs. Test a five-read single-batch alternative
to expanding the account pool, using the same frozen dense encoder and
cross-encoder. The two evaluation panels are Unix & Linux and
MathOverflow. Apply the preceding cross-site text screens. The
older [extra-read study](../crossvalidated-matched-rescue-v1)
used sequential read batches with its own fixed account sets.

For each request, score answer bodies and identify the strongest
answer per author. Retain the first three representatives in every
arm and allocate the final two reads as follows.

| Arm | Read allocation |
| --- | --- |
| Widen | Read the representatives of authors at ranks four and five. |
| Dense alternate | Prefer another answer from each of up to two different shortlisted authors, picked in dense order. |
| Residual alternate | For each of the first three authors, consider the next three dense answers. Prioritize answers containing high-IDF query terms that the current representative misses. Break ties by dense score and answer ID. |

The alternate policies fill unused slots with ranks four and five
when there are fewer than two alternate answers. All arms perform
exactly five different answer checks in one batch. Assign at most
one positive citation per author by verifier margin. Record the
number of distinct authors checked, known answers displaced and
added relative to widening, final known citations and all issuance.
Pair comparisons within shared-known-answer clusters. Also compare
the top results at each request's smaller common issuance count.
This matched comparison is descriptive and unlisted answers remain
unjudged.

In the separate shortlist audit, inspect every answer owned by the
first ten accounts to distinguish three failure modes: known answer
already representative, known answer only elsewhere in a selected
account's answer history, and no known author present. The audit
uses labels after the online route returns. All timing includes a
fresh request embedding, source planning and verifier inference.

## Results

Slurm array `45689` completed twice on the ws3 RTX 5090. Both runs
reproduced the published Unix widening baseline and all non-timing
counts. In Unix & Linux, 145 of 195 requests have a known answer
among the first ten account representatives. Of the other 50,
only six have a known answer elsewhere in a top-ten account's
history, while 44 have no known author in the shortlist. All 20
MathOverflow requests have a known representative in the first ten.

| Panel and arm | Known selected | Known issued | All issued | Warm P95 |
| --- | ---: | ---: | ---: | ---: |
| Unix, widen | 139 | 99 | 297 | 46.95–47.55 ms |
| Unix, dense alternate | 126 | 91 | 244 | 47.10–47.24 ms |
| Unix, residual alternate | 126 | 91 | 245 | 47.47–47.53 ms |
| MathOverflow, widen | 20 | 13 | 13 | 47.88–48.37 ms |
| MathOverflow, dense alternate | 19 | 13 | 13 | 47.37–48.01 ms |
| MathOverflow, residual alternate | 19 | 13 | 13 | 47.61–48.04 ms |

On Unix, each alternate route adds two known answers to its read set
and displaces 15. Each checks 618 distinct author–request combinations
compared with 975 for widening. The known-supported P@5 change is
−0.82 percentage points (paired 95% interval −1.43 to −0.31).
Matching issuance per request changes the difference to +0.21
points (−0.40 to +0.80). MathOverflow has the same known issuance
across arms. The residual terms do not improve results relative to
dense alternate selection on these panels.

These comparisons support widening as the stronger default when
only one directly supporting answer is required and author recall is
the main limitation. The experiment assigns no independent labels
to other returned views. On requests requiring separate evidence for
several qualifications, a same-author additional source can play a
different role.

[Aggregates for both runs](results.json).

## Reproduce

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.community.first_query_alternate_witness \
  CV_2024 UNIX MATHOVERFLOW QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/community-read-budget-v1/results.json NEW_RESULTS
```

Use the same frozen snapshots and checkpoints as the two previous
community studies. Repeat with `--order-offset 1`.
