# Compare role definitions before joining contributors

Supply the designated two sources from the
[engineering source panel](../engineering-cross-source-search-v1).
Each request supplies one activity phrase per source, from the NISO
examples. A predicted role applies only to its source. Candidate
selection intersects the people whose declarations support the two
selected roles. The model never sees author identities or declared
role counts while it interprets the activity. This prevents the
role scorer from favoring a plausible but incorrect interpretation
just because it yields an eligible person.

Compare six routes through the same two-clause role resolver.

- **Ridge and prototype:** take the top role from a frozen embedding
  projection for each clause.
- **Shortlist comparison:** take the union of the two best roles
  from each projection per clause. Batch the activity–definition
  pairs through the open-weight Qwen3-Reranker-0.6B.
- **Full comparison:** score every definition per clause, without
  dense shortlisting.
- **Ridge and prototype bands:** reuse the frozen single-role
  policies and require the declared roles to cover the entire band
  on each source.

For each single-role route, abstain when the smaller of the two
top-versus-runner-up score gaps falls at or below its cutoff. Fit
the cutoff separately for each route to maximize exact proofs
under a 5% adverse case cap. The calibration material consists of
14 original role descriptions rendered in 48 cases over three PLOS
paper pairs. The engineering panel uses independent NISO activity
wording on four held-out paper pairs. Repeated roles and sources
limit the independence of both samples. Details on the reused
fixture are in the [calibration benchmark](../engineering-joint-calibration-v1).

## Results

A strict proof retrieves the exact contributor set on a positive
and checks both intended declared roles. An adverse result has
any unsupported contributor or issued role. A swap output is
one type of adverse result. The role-slot column counts inclusion
before confidence gating and before matching contributors. Bands
can include several roles in one slot.

| Route | Intended role slots / 60 | Strict proofs / 15 | Adverse / 30 | Swap outputs / 15 | Mean definition comparisons | Role stage P95 across runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge top-one with cutoff | 28 | 2 | 7 | 2 | 0 | 9.8–10.4 ms |
| Prototype top-one with cutoff | 32 | 0 | 0 | 0 | 0 | 9.8–10.4 ms |
| Shortlist comparison | 36 | 4 | 8 | 2 | 4.4 | 20.4–20.9 ms |
| Full comparison | 28 | 1 | 2 | 1 | 28 | 37.4–37.5 ms |
| Ridge band | 58 | 1 | 0 | 0 | 0 | 9.8–10.4 ms |
| Prototype band | 44 | 4 | 0 | 0 | 0 | 9.8–10.4 ms |

Shortlist comparison correctly selects more intended role slots
than either embedding top-one decision. Its exact author-set
count is 10 of 15, but only four such outputs check the two
intended roles. It ties the prototype band on strict proofs,
issues eight adverse results, and roughly doubles P95. Scoring
all roles costs more and recovers fewer strict proofs. The
frozen prototype cutoff rejects all target requests; its zero
adverse results come with zero coverage. These tests support
keeping the prototype band as a low-cost baseline and measuring
a candidate method by strict proofs and adverse outcomes together.
The gap from the ridge band's 58 included role slots to one
strict proof motivates testing which ambiguous role can be
resolved from better-scoped text before strengthening the
conjunctive rule.

Both tasks in Slurm array 45913 on ws3 RTX 5090 reproduce all
non-timing fields with opposite request order. The role-stage
timer includes live embedding when needed, all comparisons in
one reranker batch, contributor intersection and serialization.
It excludes source retrieval, query parsing and network transport.
These are resident-model paths timed by component, with the
shared embedding cost attributed to each route that uses it.
[Full results](results.json)

## Reproduce

Restore the engineering panel and PLOS projection using the
linked benchmarks. Use the pinned Qwen embedding and reranker
checkpoints recorded in the results. Run inside a Slurm CUDA
allocation with a fresh output path and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.engineering_role_comparison \
  ENGINEERING_PANEL PLOS_PROJECTION NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
