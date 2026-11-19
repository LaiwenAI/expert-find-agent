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

## Stronger fixed control and exact scaling bound

The hindsight minimum can be computed directly. A returned expert requires the union of source prefixes through the first positive witness of every criterion. An excluded expert above the cutoff requires reading all witnesses for one fully failed criterion. The cheapest such criterion is sufficient. Sum these obligations in the fixed expert order and stop at the last confirmed expert, or inspect the full pool if fewer than K qualify. Source reads are keyed by expert and source. The calculation assumes any unread witness can still support its declared criteria. The direct minimum matched subset enumeration in 330 small randomized comparisons, including unavailable evidence.

The adaptive policy prioritizes refuting the criterion with fewest unread sources, then prefers a source shared by more current criteria. The fixed shortest-first control sorts criteria once by initial width, then by source sharing. Both preserve exhaustive ranked citations and reach the minimum of 605 reads across 432 publisher cases.

| Synthetic Top-3, 1,000 experts | Hindsight minimum | Fixed bundled | Fixed shortest-first | Adaptive refutation-first | Fixed / adaptive mean CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal widths, 3% support | 80,003 | 81,605 | 81,605 | 80,993 | 7.07 / 23.75 ms |
| One narrow criterion, 3% support | 21,695 | 69,376 | 21,899 | 21,761 | 4.11 / 9.27 ms |
| One narrow criterion, 35% support | 697 | 1,176 | 724 | 726 | 0.11 / 0.22 ms |

Counts sum 20 seeded pools per condition. Mean time is controller-only on ws6 with in-memory, independently generated labels and one rotated timing per arm and pool. Width-aware fixed order accounts for most of the reduction in reads. The adaptive variant saves fewer than 1% of reads over that control in the rows with gains and has higher controller cost. Net savings with a live verifier depend on its marginal source-read cost and batching behavior. [Aggregate scheduling results](scheduling-controls.json)

Necessary probing for ranked expensive predicates is established prior work. This test adds an explicitly ordered citation obligation and a read that can jointly reveal several criteria for an expert. The hindsight bound holds for a fixed indexed pool and stable Boolean observations. [SIGMOD 2002: Minimal Probing](https://doi.org/10.1145/564691.564731) · [SODA 2014: Stochastic Boolean Function Evaluation](https://doi.org/10.1137/1.9781611973402.107)

## Reproduce

```sh
PYTHONPATH=src:. python -m experiments.credit.local_constraint_certificate \
  benchmarks/plos-credit-compositional-edits-v1 /path/to/new-results.json
PYTHONPATH=src:. python -m experiments.credit.constraint_read_lower_bound \
  benchmarks/plos-credit-compositional-edits-v1 /path/to/new-lower-bound.json --stress
```
