# Certify shallow planning against a fixed depth-128 reference

Reuse the 452-article [mixed publisher catalog](../mixed-publisher-joint-v1)
and its 211 queries. Every route performs live hybrid retrieval to
depth 128 for both requested topics. The adaptive route initially
projects personal-role postings only for ranks 32 or 64 per topic.
The fixed reference projects all 128 from the beginning and opens
the three nominated declarations.

The adaptive route applies this rule:

1. Find the shallow anchor pair using the published personal-role
   selection key. Proceed with two actual source reads only when
   the pair has both indexed roles for the same contributor. The
   primary key favors fewer unknown roles, then the worse topic
   rank and rank sum. A deeper pair that includes a source outside
   the shallow prefix for its condition has a worse rank, so a
   fully indexed shallow anchor pair remains the deep anchor pair.
2. Verify the opening two declarations. Stop only when an actual
   full proof exists and no unread shallow source could combine
   with observed evidence to outrank that proof, even after
   optimistically granting any role to every shared author.
   All evidence that lies below the shallow depth for either
   condition has a worse rank than the verified proof.
3. Otherwise complete the depth-128 plan. Reuse the shallow
   postings and any declarations already opened. The adaptive
   path reads at most three distinct sources, including fallback.

This certificate relies on complete bylines for the available identity
inventory, frozen ranks, two distinct sources, and the specified
deterministic proof order. Indexed roles can be wrong or missing.
The adaptive route still checks original declarations. It makes no
guarantee about broader inventories or unobserved contributors.

Evaluate unavailable role fields, three deterministic 50%-coverage
masks, and controlled rotations of roles between identified authors on
the same article. Exactly compare the returned person and both citations
with the fixed-depth output per request. A foreign publisher result
with no aligned label remains unjudged. A projection count is the
number of document positions converted into query-specific role
facets. It excludes the optimistic proof check, raw title and abstract
ranking and model inference, so it does not estimate total work.

A separate warm timing pass runs all policies from JSON parsing through
response encoding for clean full fields and one 50%-coverage
rotation mask. It includes live query encoding and hybrid ranking,
posting projection and three or two actual local declaration reads.
Each comparison uses identical request input and checks exact
serialized response parity. Remote I/O, queues and index construction
are outside the timing boundary.

## Results on RTX 5090

Both tasks of Slurm array `44631` on ws5 reproduced all non-timing
results. Each shallow route matches the fixed-depth person and
citations on all 2,321 request-settings per run. The exact agreement
also holds for the 1,500 generated examples used to check the stopping
rule with independently masked and incorrect indexed role fields.

With complete, unperturbed publisher fields:

| Panel and starting depth | Certified at shallow depth | Source reads | Projected positions | Live local P95 |
| --- | ---: | ---: | ---: | ---: |
| PLOS fixed 128 | 0 | 480 | 40,960 | 10.33–10.35 ms |
| PLOS adaptive 32 | 56 | 424 | 30,208 | 10.32–10.43 ms |
| PLOS adaptive 64 | 99 | 381 | 28,288 | 10.32–10.38 ms |
| eLife fixed 128 | 0 | 153 | 13,056 | 11.18–11.27 ms |
| eLife adaptive 32 | 15 | 138 | 10,176 | 10.94–11.21 ms |
| eLife adaptive 64 | 18 | 135 | 10,752 | 11.17–11.27 ms |

At 64, the adaptive route saves 117 of 633 source reads over both
panels and retains the fixed-width coverage of 85 known PLOS and 46
known eLife requests. Projection cost and the number of deferred
reads vary by starting depth. Warm local P95 remains similar because
the ranker still computes all 128 candidates.

At 50% clean exposure, adaptive 64 certifies only 15–50 PLOS and
3–4 eLife requests across the hash masks. Under the rotation
intervention at the same exposure, it certifies 15–50 and 1–4,
respectively. Ranges across these masks are descriptive. At zero
indexed coverage both panels always expand. The adaptive guarantee
still holds at those settings because it is relative to the same
fixed-128 index, including its missingness or errors. It does not
repair that index's mistakes.

## Reproduce

Restore the pinned publisher snapshots and embedding checkpoint from
the live request benchmark. Run twice on a Slurm GPU, using a fresh
result path and `--order-offset 1` for the second run.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.adaptive_role_depth \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json NEW_RESULTS
```

[Replicated results and timing ranges](results.json).
