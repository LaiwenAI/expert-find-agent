# Compare enterprise overlap review schedules at a fixed pair budget

Evaluate the 110 new representatives from the
[regional frame](../enterprise-regional-frame-v3) against the
245 earlier representatives, other additions, and 4,301 historical
demands. Reuse first and last windows and the same frozen checkpoint as
in the [overlap screen](../enterprise-regional-overlap-v1). Encode
identical window strings once and compare the following policies for each destination.

- **Stock.** Use the existing high-score ordering with eight
  review slots. When no pair meets the threshold, rank the strongest
  nearest neighbors across the focal groups.
- **Distributed.** Use the top four neighbors per focal group and
  destination. At each step prefer a focal endpoint without an
  existing review, then a less represented publication source.
  Within that order prefer pairs that cover an additional focal
  endpoint and then higher cosine. Carry focal coverage across
  destinations. Use eight pairs per destination.

Report distinct focal groups and their publication-source counts.
When a pair connects two additions, both focal endpoints count.
The compare-and-review packet unions both policies and both encoding
orders. A blinded form omits policy, rank, score, and destination
labels. Keep the union, mapping, and input projections private.

The separate known-source stress test mixes 54 title-free central
excerpts with the 56 remaining title-free full bodies from the
[controls](../enterprise-overlap-controls-v1). Each query contributes
its rank-one candidate. Compare ranking that queue by global score
with round-robin prioritization by publication source, preserving
score order within each source. Record the known-source total and
short-source contribution at budgets of 8, 24, and 48. This tests
allocation among queries and leaves the candidate scorer unchanged.

## Results

Both jobs in Slurm array 46217 completed on ws3 RTX 5090 in
37 seconds. Both orders selected identical pairs and reproduced
the previously archived stock semantic selection.

| Policy | Pairs | Distinct focal groups | Jiangyan | Liaocheng | Zhangjiagang | Zibo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock | 24 | 27 | 9 | 2 | 13 | 3 |
| Distributed | 24 | 32 | 11 | 7 | 8 | 6 |

Both policies submit a 24-pair queue. Their union contains 34 distinct
pairs, including 24 from the original review packet and 10 additions.
Two blank reviewer copies pass integrity validation. Natural duplicate
yield and false-negative rates remain unmeasured.

| Stress-test budget | Highest score, short source | Source balance, short source | Total known sources, either policy |
| ---: | ---: | ---: | ---: |
| 8 | 1 | 2 | 8 |
| 24 | 3 | 7 | 24 |
| 48 | 10 | 14 | 48 |

Source balance yields more short probes at the same count, but the
score ordering within each source still favors full bodies. Uniform
sampling would include 11.8 short probes in expectation at budget 24
under this constructed population. That expectation concerns exposure
to a probe type, and natural duplicate yield must be compared by
judgment on the common packet. See [aggregate results](results.json).

## Reproduce

Restore the snapshots, private indexes, and both earlier projections
using the [regional screen](../enterprise-regional-overlap-v1#reproduce)
and [known-source controls](../enterprise-overlap-controls-v1#reproduce).

```sh
umask 077
# Run the following two commands inside Slurm GPU allocations.
PYTHONPATH=src:. python -m experiments.enterprise_overlap_budget gpu \
  PRIVATE_REGIONAL_PROJECTION PRIVATE_CONTROL_PROJECTION QWEN_EMBEDDING GPU_A --order 0
PYTHONPATH=src:. python -m experiments.enterprise_overlap_budget gpu \
  PRIVATE_REGIONAL_PROJECTION PRIVATE_CONTROL_PROJECTION QWEN_EMBEDDING GPU_B --order 1

PYTHONPATH=src:. python -m experiments.enterprise_overlap_budget packet \
  EARLIER_SNAPSHOTS MUNICIPAL_SNAPSHOTS REGIONAL_SNAPSHOTS PRIVATE_WORKBOOK \
  PRIVATE_245_INDEX benchmarks/enterprise-regional-frame-v3/results.json PRIVATE_355_INDEX \
  PRIVATE_REGIONAL_PROJECTION PRIVATE_CONTROL_PROJECTION GPU_A GPU_B PRIVATE_REVIEW_DIR
PYTHONPATH=src:. python -m experiments.check_enterprise_panel_review \
  PRIVATE_REVIEW_DIR REVIEWER_A_JSONL REVIEWER_B_JSONL
```
