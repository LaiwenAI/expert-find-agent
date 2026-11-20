# Pool the same 4B review budget across ready topics

Reuse the verified FollowIR score replay from the
[same-passage study](../followir-same-passage-routing-v1).
The inputs are the same 50 Robust04 topics, 575 documents, 0.6B
and 4B decisions, and cutoffs fitted on Core17 and News21. The
replay operates on the revised criterion. Published live timing
is from the earlier, separate benchmark.

The local route sends the floor of one quarter of each topic's
documents to 4B, nearest to the compact verifier cutoff first.
The pooled route treats consecutive blocks of five topics as
already ready for review. Keep their sum of local budgets and
reserve one review for every topic with positive local budget.
Merge the remaining priorities. A topic can use up to one or
two additional reviews relative to its original allocation.
Use nearest-cutoff and hash ordering as separate controls.
The 4B output replaces the compact decision on selected rows.

Rotate the public topic order by zero through four positions
before forming blocks. These deterministic shifts test sensitivity
to group composition on one shared panel. They are neither
independent replications nor traces of concurrent service arrival.

## Results

All routes send exactly 140 documents to 4B. The ranges below
span the five group compositions. Counts represent human-assessed
document relevance, not ratings of the passages themselves.

| Route | Stable positives accepted | Violations accepted | Topics with an accepted stable positive |
| --- | ---: | ---: | ---: |
| Local boundary | 15 | 5 | 11 |
| Pooled boundary, one extra | 15–16 | 5 | 11–12 |
| Pooled boundary, two extra | 15–17 | 4–7 | 11–12 |
| Pooled hash, one or two extra | 9–11 | 4 | 7–9 |

All shown routes accept zero stable negatives. The local route
shares 14 stable-positive decisions with the all-4B reference,
accepts one positive that all-4B rejects, and misses six all-4B
positives. In their respective topics, these six unreviewed
documents rank fourth, fourth, fifth, fifth, fifth and ninth
under nearest-cutoff priority. Allowing one extra per topic
makes two of the six eligible and allowing two extra makes five
eligible. They still have to beat reviews in other topics in
the same group.

Reallocating the same compute by distance to the threshold
provides at most two additional stable positives and remains
sensitive to topic grouping. The extra calls may also increase
constraint violations. The results motivate improving within-topic
review priorities before engineering a shared waiting queue.
This replay does not measure service latency or establish a
deployable policy for the enterprise setting.
[Aggregate results](results.json).

## Reproduce

Use the original scores from the pinned GPU run. The analysis
runs on CPU and checks the original compact thresholds, projection
hashes and the full local-route aggregate.

```sh
PYTHONPATH=src:. python -m experiments.followir.pooled_verifier_budget \
  SCORES_0 benchmarks/followir-same-passage-routing-v1/results.json NEW_AGGREGATE
```
