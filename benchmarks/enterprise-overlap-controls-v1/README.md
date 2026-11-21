# Test enterprise overlap sensitivity with known-source excerpts

Reconstruct the frozen [regional overlap projection](../enterprise-regional-overlap-v1)
and require the first and last windows of all 4,656 demand records to
match. Use the same frozen Qwen3-Embedding-0.6B checkpoint, 384-token cap,
and maximum cosine across query and corpus windows. Compare three corpus representations
with the same encoder. Use first and last, first and center, or all three. Each window retains the original corpus title.
Identical window strings are encoded once.

- Remove the title from all 110 new regional demands. For bodies of
  at least 200 characters, also query with the central 180 characters.
- Use up to 80 historical demands per location condition, sampled
  deterministically with one demand per normalized issuer. Require a
  body of at least 1,460 characters and a central 180-character span
  absent from its first and last 640 characters. Probe the same
  source with 180 characters from the head, center, and tail.
- Compare recovery of the exact originating record at each threshold
  and its rank among all 4,656 records. Count other records above
  threshold as review load. They may include genuine neighbors.

The controls use original text excerpts and remove the title. Naturally
rephrased duplicates and unobserved issuer aliases require separate
evaluation. No expert-search quality labels are involved.

## Results

Both jobs in Slurm array 46212 completed in 48 seconds on ws3
RTX 5090. Reversing the encoder's batch order moved some counts
by one. Intervals in the table give the two observed counts and
carry no sampling-confidence interpretation.

| Known-source probe | First and last at 0.85 | First and center at 0.85 | All three at 0.85 | First and last top 1 |
| --- | ---: | ---: | ---: | ---: |
| Regional body without title | 104/110 | 104/110 | 104/110 | 110/110 |
| Regional central excerpt without title | 32/54 | 32/54 | 32/54 | 54/54 |
| Historical start excerpt without title | 23/80 | 23/80 | 23/80 | 76/80 |
| Historical hidden central excerpt without title | 5/80 | 25/80 | 25/80 | 63/80 |
| Historical tail excerpt without title | 10-11/80 | 7/80 | 11-12/80 | 58/80 |

A higher cutoff of 0.90 recovers only 16/54 regional central excerpts
and 2/80 hidden historical centers with first and last windows. On the
regional central probes, 0.80 recovers 44/54 at that same window
configuration. Rank and threshold therefore measure different losses.
The earlier zero-hit audit gives weak evidence about short excerpts
at the fixed threshold.

Replacing the last window with the center helps central excerpts
and loses tail hits at essentially identical index size, 7,593 versus
7,594 distinct corpus strings. Adding the center instead raises the
index to 10,531 strings (+38.7%) and gives the same threshold recovery
for central excerpts. At 0.85 it yields one additional record above
threshold for the historical center probes. That is an observed
review load in the constructed controls. Precision and added workload
for natural duplicate search remain to be measured.

None of the 110 new regional representatives meets the long-body
criterion, so the hidden-center result applies to the historical long
stratum. A follow-up can test a per-group nearest-neighbor fallback
and selective center coverage under an explicit pair-review budget.
Keep the original audit thresholds and the provisional frame frozen
until real candidate pairs receive independent judgments.
See [both aggregate runs](results.json).

## Reproduce

Use the snapshots, private indexes, and GPU projection from the
[regional overlap audit](../enterprise-regional-overlap-v1#reproduce).

```sh
umask 077
PYTHONPATH=src:. python -m experiments.enterprise_overlap_controls prepare \
  EARLIER_SNAPSHOTS MUNICIPAL_SNAPSHOTS REGIONAL_SNAPSHOTS PRIVATE_WORKBOOK \
  PRIVATE_245_INDEX benchmarks/enterprise-regional-frame-v3/results.json \
  PRIVATE_355_INDEX PRIVATE_REGIONAL_PROJECTION PRIVATE_CONTROL_PROJECTION

# Run the following commands inside Slurm GPU allocations.
PYTHONPATH=src:. python -m experiments.enterprise_overlap_controls gpu \
  PRIVATE_CONTROL_PROJECTION QWEN_EMBEDDING PRIVATE_RESULT_A --order 0
PYTHONPATH=src:. python -m experiments.enterprise_overlap_controls gpu \
  PRIVATE_CONTROL_PROJECTION QWEN_EMBEDDING PRIVATE_RESULT_B --order 1
```
