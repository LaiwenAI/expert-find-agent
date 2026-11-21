# Frozen first-query benchmark suite

The suite separates measured evidence coverage, exact execution, and
independently judged expert relevance. Its [protocol](protocol.json)
pins datasets and source projections through the original archive and
input digests. Each runner keeps the same candidate menus, read and
check budgets, judges, and response format for the paired comparison.
Results stay separate by panel and judgment type.

| Track | Fixed dataset and task | Registered controls | Metrics |
| --- | --- | --- | --- |
| Open-topic nomination | 160 PLOS and 51 eLife requests over 452 papers. Three reads per request. | Independent anchors plus same-role sequential repair against static role-aware joint nomination. Both see the same role postings. | Publisher-tag-supported proof count, natural and matched issuance, source reach, paired gains and missing-label bounds. Frozen quotas are 45 PLOS and 3 eLife returns. Warm local P95. |
| Exact role index | Independent extended and confirmation eLife projections. 205 and 193 requests across 2,011 sources. Original and connected subcatalogs at depths 32 and 128. | Full roles, globally linked records, query-local links, packed repeated-author roles, packed roles plus query-local links. | Exact ordered read, expert, citation and abstention parity. Role bytes, adjacency bytes, role rows, planning P95 and local request P95. |
| Expert and citation | 49 TREC Enterprise 2006 expert topics with historical owner and document menus. | Reuse-first, fitted dispatch on the original shortlist, fixed plan, expanded nominees, and joint or adaptive scheduling variants. | A supported slot requires grade-2 expert relevance and grade-2 document support. P@5 counts empty slots as zero. The scorecard retains paired intervals, observed reads, checks, and a judgment-aware read ceiling. |
| Reviewer relevance | CoF KDD (122 requests) and CoF SciRepEval (59), fixed top-five candidates and captured source decisions. | Dense scheduling, learned acceptance, rank prior, observed relevance weighting, judged relevance weighting. | Known relevant, judged negative and unjudged among matched issued packages. The frozen totals are 288 and 99 cited slots. Paired P@5 differences, missing-label bounds, reads and tokens. |
| Enterprise diagnostic | 123 frozen enterprise groups with fixed corpus, checkpoints, and original-text scope. | Dense and hybrid rankers, with no rescue, shortlist rescue or full-index rescue. The primary comparison uses the two dense rescue arms. | Independently adjudicated supported P@5, coverage, adverse risk, checks, serial P95 and paired intervals. |

The open-topic labels indicate publisher-supported evidence. CoF labels
reviewer relevance and has no independent citation judgments. Neither
track substitutes for the enterprise acceptance test. The enterprise
diagnostic is tied to its [frozen scoring protocol](../enterprise-sealed-first-query-v1).
It cannot establish the full system's deployment claim. A new independent
cohort must separately verify supported P@5, client P95, cost per
supported success, and the absolute risk and coverage gates. Its
sampling frame is [provisional](../enterprise-public-compendium-v4).

Published methods serve as controls when they match the task and
information budget. The current tracks reuse hybrid retrieval, static
role projection and learned dispatch. Reviewer-profile assessment from
[MERIT](../merit-author-assessor-v1) and the released
[RATE comparisons](../merit-author-retriever-v1) remain separate
reference assays with their original cohorts and inference settings.

## Run and refresh the scorecard

The [reference scorecard](reference-scorecard.md) is reproduced by the
collector without downloading models or reading private records.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.first_query_suite collect \
  scorecard.json --markdown scorecard.md
```

To test a new strategy, implement its policy in the corresponding
registered runner. Retain the frozen comparison arms and inputs.
The suite runs the track in both orders, checks the result format and
input fingerprints, and produces a paired result. Run the track inside
a Slurm compute allocation.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.first_query_suite run \
  trec new-trec.json TREC_QRELS_DIRECTORY
PYTHONPATH=src:. python -m experiments.retrieval.first_query_suite collect \
  updated-scorecard.json --result trec=new-trec.json --markdown updated-scorecard.md
```

The TREC run reproduced all nine arms' non-timing results in both
orders on ws6 (Slurm job 47560). The other reference panels were
collected from their pinned experiment archives. The input guard
rejected a TREC replay with changed input bytes.

The other track input orders follow their original commands, omitting
their output argument.

- `open_topic` uses the [matched-role inputs](../open-topic-role-matched-v1#reproduce).
- `linked_role` uses the [two independent projections](../linked-role-lane-index-v1#reproduce).
- `cof` uses the [two captured relevance panels](../cof-relevance-weighted-routing-v1#reproduce).

The scorecard retains the frozen and candidate results together.
Changed input fingerprints, unmatched issuance quotas, changed
control metrics, and decision differences between replay orders
stop collection. Existing archived paths and signatures stay pinned.
For the private enterprise diagnostic, the collector accepts the
aggregate produced by the registered scorer through
`--enterprise-result` after blind review and reconciliation.

The `run` command executes benchmark-specific policy entry points.
Each benchmark still owns its dataset preparation and scorer.
