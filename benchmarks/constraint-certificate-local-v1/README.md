# Candidate-local stopping for ranked hard constraints

A fixed expert prior lets the certificate process one candidate at a time. For each criterion, a cursor points to the first witness that is unread or positively supported. Exhausting a cursor rules out the candidate. Confirming every cursor certifies the candidate and its canonical citations. Otherwise the controller reads the most frequent unresolved source on the frontier. It stops after certifying the requested number of experts or exhausting the pool.

The whole-pool controller chooses the earliest possible candidate with an unresolved frontier among its optimistic Top-K. Confirmed candidates preceding it are already fixed. The local controller examines the same candidate and frontier without rebuilding either ranked bound across the pool. Both choose the same source on ties. This establishes identical actions for a fixed order and non-retracting Boolean source judgments. The proof does not extend to dynamic reprioritization or a new source retracting previously trusted support.

The timing replay rotates four strategies on Apple M4, with seven observations per request and quantiles over per-request medians. The 1,000-expert setting uses one timing observation per trial. Timing begins with an ordered candidate pool in memory and reads declaration judgments from in-memory dictionaries. It excludes retrieval, text decoding, model inference, network and storage I/O.

The publisher cohort reuses the 108 test views from [the PLOS ONE compositional fixture](../plos-credit-compositional-edits-v1/README.md) at both Top-1 and Top-3 and both scope settings. The synthetic stress cohort takes 20 seeded pools per condition at 10 and 100 experts and five pools per condition at 1,000 experts. Each has four constraints, six candidate sources per expert, and four allowed witnesses per constraint. It isolates controller scaling with independent support probabilities of 3% and 35%. It gives no estimate of realistic long-tail support prevalence.

Every run checks expert and citation equality against exhaustive evaluation and checks exact action-trace equality between the two certificate implementations. Fixed ordering by bundled source coverage is a separate control. The certificate is an instantiation of monotone stopping for a ranked witness output. See [Fagin et al. on threshold algorithms](https://research.ibm.com/publications/optimal-aggregation-algorithms-for-middleware) for prior work on bounded Top-K access.

## Why the fixed schedule ties on publisher data

An oracle enumerates all subsets of reads and finds the smallest subset that certifies the exhaustive output for each case. It knows all answers but checks certification with only the selected answers exposed. Both the fixed schedule and the local certificate attain this lower bound in all 432 query, scope and cutoff combinations, with 605 reads each in total. This rules out further scheduling-only read savings on this fixture under the same independent unknown-witness and source-read contracts. [Oracle counts](oracle-read-lower-bound.json)

## Timing and scaling

| Cohort | CPU P50, whole-pool certificate | CPU P50, local certificate | CPU P50, fixed bundled | Mean reads, local / fixed bundled |
| --- | ---: | ---: | ---: | ---: |
| Publisher data, specified source, Top-1, both sources present | 10.21 us | 4.88 us | 4.60 us | 1.78 / 1.78 |
| Synthetic, 100 experts, 3% support | 44.34 ms | 0.81 ms | 0.45 ms | 425.90 / 407.85 |
| Synthetic, 1,000 experts, 3% support | 4,058.55 ms | 8.28 ms | 4.50 ms | 4,254.20 / 4,079.80 |
| Synthetic, 1,000 experts, 35% support | 10.41 ms | 0.05 ms | 0.05 ms | 29.20 / 28.60 |

The candidate-local state removes repeated whole-pool work. The fixed bundled control remains faster in the sparse stress setting and reads fewer sources there. Gains on top of that control would require a better action order on a more discriminating fixture, cheaper judgments, or an index that rules out candidates with admissible evidence. [Timing aggregates](results.json)

## Reproduce

```sh
PYTHONPATH=src:. python -m experiments.credit.local_constraint_certificate \
  benchmarks/plos-credit-compositional-edits-v1 /path/to/new-results.json
PYTHONPATH=src:. python -m experiments.credit.constraint_read_lower_bound \
  benchmarks/plos-credit-compositional-edits-v1 /path/to/new-lower-bound.json
```
