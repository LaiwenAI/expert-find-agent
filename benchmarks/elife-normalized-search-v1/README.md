# Normalize query duties once in a live eLife expert search replay

Use the pinned
[eLife article projection](../elife-role-nomination-v1) with
the 280 sources that have complete, parseable author declarations.
Map the five earlier publisher roles to the declared eLife CRediT
fields: Methodology, Investigation, Formal analysis, Writing –
original draft, and Resources. This is a limited role contract.
It does not cover all CRediT fields. For the keyword topics that occur
in at least four articles, take the same one rare and one frequent
construction, restricted to the five field pairs. The query includes
a topic string and two authored duty phrases from the
[PLOS normalization study](../plos-study-role-normalization-v1).
Replace one duty with an outside-schema phrase for a matched control.
The topics and declaration format are new relative to the PLOS test.
The authored phrases are reused.

Use the same dense/lexical Qwen3-0.6B topic ranking, eight-source read
budget, relevance checker and cutoff in every arm. Verify role
eligibility on articles passing the topical cutoff. De-duplicate
authors across sources when the catalog has a stable identity.
Keep up to five people with a supporting article. Compare a direct
Qwen3-4B binary decision per author against the same agreement-based
normalizer run once per source and once per request. Exact owner
and role matching follows normalization in both normalizer arms.
Document fields stay outside the query interpretation model.

Measure from the supplied topic/role query to the returned ordered
recommendation list, with query encoding, ranking, topical
verification, normalization and checking all live. Warm each arm
and rotate its order across queries. There is no cross-query inference
cache. The timings exclude input extraction from unstructured demands,
index construction, service queues and external input/output. The
role validity endpoint is the chosen declaration contract. A tag-backed
slot requires both valid role assignment and an author keyword match.

## Results

Both tasks in Slurm array 44174 ran on an RTX 5090. Counts agree
in both rotated orders. The panel contains 18 topics and 36
covered requests with one paired outside-schema request each.

| Arm | Covered valid role slots / issued | Covered tag-backed slots | Covered serial P95, ms | Outside-schema slots issued | Outside serial P95, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct binary, per author | 43 / 166 | 22 | 468.9–471.8 | 164 | 545.9–548.7 |
| Normalize per source | 96 / 96 | 58 | 267.2–270.1 | 0 | 314.1–316.2 |
| Normalize per request | 96 / 96 | 58 | 85.2–85.5 | 0 | 84.4–84.5 |

The two agreement normalizers produce identical ordered authors
and source references on every query. Across the covered requests,
they use 107 and 34 role-model forwards respectively. The same
counts on outside-schema requests are 132 and 34. The warm local
request P95 decreases by about 68% for covered queries and 73%
for the outside-schema controls. These speed comparisons share
the same role logic and prompts. The direct baseline uses the
affirmative digit 1, a separate operating point with a known
sensitivity to output encoding.

Complete indexed declarations were available by construction.
The study does not measure whether automatically extracted role
claims are sound in noise or free prose. The output evidence is
checked against the publisher fields. Topic adequacy is still a
publisher keyword proxy. The outside-schema set recycles authored
duties and does not estimate open-world risk.

[Both Slurm replays](results.json)

## Reproduce

Rebuild the eLife projection from its pinned manifest. Run with
the verified Qwen checkpoints from the prior studies inside an
allocated Slurm GPU, using this repository version for all imported
modules.

```sh
export PYTHONPATH=src:.
python -m experiments.credit.elife_role_normalization_search PROJECTION \
  benchmarks/elife-role-nomination-v1/manifest.json \
  benchmarks/elife-role-nomination-v1/results.json NEW_RESULTS --order-offset 0
```

Repeat with order offset 1 and a fresh output path.
