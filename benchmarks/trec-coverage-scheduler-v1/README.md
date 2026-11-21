# Schedule document reads for distinct supported experts

Reuse the [expert-first candidate and source menus](../trec-nomination-waterfall-v1)
on 49 judged TREC 2006 Enterprise topics. Each method has the same
20 candidates, first 16 sources per candidate, and limit of 13
candidate–document checks. For each topic, take the actual distinct
document count of the prior document-first reference as the read cap.
The policy does not change nomination or candidate order.

Train a small ridge-regularized logistic model separately for every
held topic using only the other 48 topics for supervision. Remove the
held topic's judgments from all training feature graphs and mask each
training topic's own judgments while constructing its menus. The
features are candidate rank, source rank, candidate history size,
document fan-out, number of historical support links, accumulated
topic votes and the strongest lexical topic similarity of a link.
A positive training pair requires grade 2 for both expert and
candidate-specific document support. Other pairs supply zero observed
credit, including unjudged pairs. Thus the target estimates **recoverable
judged support** in this archive; calibration here gives no reliability
claim for missing labels or deployed verification.

| Read schedule | When no opened source has another eligible check |
| --- | --- |
| Reuse reference | Score unseen sources by sum of discounted candidate and source ranks. |
| Fitted greedy | Choose the source with largest summed estimated support for unresolved experts. |
| Fitted lookahead | Evaluate pairs among the first 24 sources by estimated distinct-expert coverage, take the first read of the strongest pair, and replan after checks. |

The lookahead formula is the sum over owners of the estimated probability
of at least one supporting citation, assuming independence of the two
source outcomes. This is related to intent coverage in prior search
diversification work. Here the covered items are named experts and
reads are followed by separate verification of each expert–document
pair. The independence assumption can fail for correlated evidence.
[Santos et al., SIGIR 2011](https://doi.org/10.1145/2009916.2009997)

All schedules reuse an opened document when it has another eligible
check and stop checking an expert after a grade-2 document. Held
document labels are revealed only after choosing a check. Report
realized work, unknown checks, issued slots and supported P@5. Topic
bootstrap intervals compare the two fitted routes to the reuse
reference. The earlier oracle ceiling uses held labels and is a
diagnostic upper limit rather than a feasible deployment method.

## Results

| Policy | Distinct query–document reads | Association checks | Supported slots |
| --- | ---: | ---: | ---: |
| Reuse reference | 103 | 285 | 9 |
| Fitted greedy | 103 | 383 | 11 |
| Fitted lookahead | 103 | 384 | 11 |

All policies share the read cap and the limit of 13 checks per
topic. The learned variants use more of the available checks. Both
give the same supported P@5 change of +0.82 percentage points,
with paired 95% interval −1.22 to +3.67. Both issue 12 slots, one
of which lacks grade-2 expert relevance. The reference issues
9, all known positives. The learned Brier score for observed support
over 15,398 held pairs is 0.01124, versus 0.01079 for the
per-training-set positive-frequency predictor.

The second diagnostic gives each policy no more than the reference's
realized reads **and** realized checks for that same topic. It was
added after inspecting the realized effort difference in the first
comparison. It reuses the fitted model, menus and feedback rule.

| Policy under realized reference caps | Reads | Checks | Supported slots |
| --- | ---: | ---: | ---: |
| Reuse reference | 103 | 285 | 9 |
| Fitted greedy | 80 | 266 | 10 |
| Fitted lookahead | 81 | 266 | 10 |

Both learned routes have a paired supported P@5 change of +0.41
percentage points (95% interval −1.63 to +3.67). The greedy policy
reads 23 fewer documents and makes 19 fewer checks. It issues one
additional slot lacking grade-2 expert relevance. Lookahead provides
no measured benefit over greedy selection. This is a partial-label
archive replay and does not measure document I/O or entailment
model latency. Keep reuse as the comparison route and treat fitted
greedy as a candidate for independent evaluation.

Both Slurm CPU replicas agree on all results. The stricter rerun
also reproduces the original common-cap and calibration results.
[Aggregates](results.json)

## Reproduce

Restore the [official topics and qrels](https://trec.nist.gov/data/t15_enterprise.html)
and use a Slurm CPU allocation. Repeat with reverse topic order.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m experiments.trec_enterprise.nomination_coverage_scheduler \
  TREC_DATA NEW_RESULTS
```
