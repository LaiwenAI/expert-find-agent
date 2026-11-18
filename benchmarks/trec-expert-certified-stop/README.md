# Exact stopping for evidence-ranked experts

The top five ranks are sorted by confirmed status, then by a frozen expert prior. Confirmation requires one or two accepted sources. The source order within each expert is frozen and the output attaches the first sources that meet that count.

Once five experts are confirmed, an unresolved expert below the fifth confirmed expert in prior order cannot displace any of the current five. Only unresolved experts above that boundary with enough remaining sources can change the result. The boundary can only move to a better prior rank. If fewer than five are confirmed, every unresolved expert still capable of meeting the source count remains eligible. This certificate applies to the fixed ranking rule and simulated verifier outcomes. It is a bounded Top-K stopping control related to the [threshold algorithm](https://research.ibm.com/publications/optimal-aggregation-algorithms-for-middleware).

## Pooled replay

Use the 49 judged [TREC 2006 Enterprise topics](https://trec.nist.gov/data/t15_enterprise.html), ten candidates per topic from a leave-topic-out prior, and 12 paired source permutations. Compare each schedule with and without the certificate. All 14,112 certified runs match the exhaustive reference's ordered expert IDs and attached source IDs across the three verifier profiles and two support counts.

The table reports averages for the independent simulated verifier with sensitivity 90% and false positive rate 1%. A wave checks up to four sources for different experts and has no wall-clock interpretation.

| Required accepted sources | Schedule | Checks before → after | Unique reads before → after | Waves before → after |
| --- | --- | ---: | ---: | ---: |
| 1 | Serial prior order | 186.23 → 153.23 | 167.61 → 138.99 | 186.23 → 153.23 |
| 1 | Batched prior order | 186.23 → 163.04 | 167.61 → 147.41 | 63.72 → 54.84 |
| 1 | Batched round robin | 186.23 → 172.44 | 167.61 → 155.57 | 62.06 → 57.49 |
| 2 | Batched prior order | 250.42 → 238.63 | 220.67 → 210.74 | 81.02 → 76.18 |
| 2 | Batched round robin | 250.42 → 244.57 | 220.67 → 215.61 | 78.18 → 76.14 |

With one accepted source required, batched prior order saves 23.20 checks on average and reduces P95 check count from 429.80 to 402.90. Requiring two sources narrows the average savings to 11.79 checks. The certificate preserves the selected citations even when the simulated verifier accepts a non-supporting source. Pooled associations, simulated responses, and a frozen prior limit this result to verification work. Full-corpus retrieval and live model latency are outside this study.

## Schedule multiple checks for the highest-ranked unresolved expert

A wave can spend several of its four slots on the next consecutive sources of one expert. Commit decisions in the frozen source order. If an early decision confirms or eliminates an expert, ignore the later decisions for ranking and still charge all scheduled checks and document reads. Recompute the stopping boundary between waves.

The new replay checks all six schedules across the same topics, source orders, verifier profiles and support counts. All 21,168 runs reproduce the exhaustive reference's ordered expert and source IDs. The table reports simulated averages with independent noise and one required source.

| Schedule | Expert–source checks | Unique reads | Waves |
| --- | ---: | ---: | ---: |
| One per expert, prior order | 163.04 | 147.41 | 54.84 |
| Prioritize cached sources | 163.13 | 147.46 | 54.86 |
| Group experts by next source | 163.13 | 147.46 | 54.87 |
| Group by source and decision impact | 163.17 | 147.50 | 54.89 |
| Up to two consecutive sources per expert | 158.64 | 143.62 | 42.94 |
| Up to four consecutive sources per expert | 159.57 | 144.51 | 40.00 |

Two-source batches reduce average waves by 22% and unique reads by 2.6% in the independent-noise, one-source condition. The whole-topic bootstrap interval for wave savings is 9.39 to 14.44 waves. With perfect verification and two required sources, two-source batches reduce waves from 81.59 to 70.03 and increase unique reads from 225.99 to 226.84. Larger batches amplify that read tradeoff. Prioritizing shared next sources saves no work on these pools. These are scheduled work counts under simulated verification. Live GPU latency, ownership judgments and concurrent load require a separate comparison. [Aggregate results](source-dispatch.json)

Run [the schedule comparison](../../experiments/trec_enterprise/certified_source_dispatch.py) with `QRELS_DIR NEW_RESULT_JSON`.

## Reproduce

Download `ent06.qrels.expert` and `ent06.qrels.expert-docs` from the NIST track page. Run:

```sh
PYTHONPATH=src:. python -m experiments.trec_enterprise.certified_expert_stop QRELS_DIR NEW_RESULT_JSON
```
