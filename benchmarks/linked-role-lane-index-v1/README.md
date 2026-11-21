# Transfer exact role pruning and test opposite-lane filtering

Use the pinned [extended eLife projection](../elife-extended-role-holdout-v1)
and [confirmation projection](../elife-source-guard-confirmation-v1).
The two snapshots share zero DOI or validated contributor IDs.
Use the archived heading-to-role query constructor on each snapshot
to build an open-topic panel. Publisher subject tags and declared
roles support the reference requirements. Subject tags are not
available to retrieval. The study does not judge expert suitability.

Compare the full role index with the prior exact globally linked
source index and a query-local filter. Repeat the comparison against
the stronger [packed repeated-author projection](../role-pair-placement-v1).
For source `d`, store a bit mask of the other sources sharing its
byline. On each query, construct membership masks for both ranked
lists. Check `neighbors[d] & opposite_lane` before looking up roles.
A zero intersection means the source cannot extend a distinct-source,
same-person pair across these two conditions. Apply this filter only
to the role lookup. Keep all original ranks and the same fallback.
Test 32 and 128 candidates per condition.

Replay the queries on the original snapshots and on subcatalogs that
retain every globally linked source. The latter are stress conditions
in which global link pruning retains every document. They reuse the
same requests and do not constitute additional independent samples.
Within each condition, require the full and all sparse variants to
reproduce the same ordered reads, returned person, exact citations and
abstentions. Keep three distinct reads and the original verifier.

Record role lookup work as the number of source–condition rows.
Measure both plan formation and warm local end-to-end P95. Count the
added adjacency payload as the encoded masks plus a four-byte length
per linked document. Ordinals are shared with the linked source table.
The reported sizes exclude Python overhead, shared postings, and source
documents. Timing includes live encoding and local reads and excludes
network wait or queueing.

## Results

Slurm job 47558 completed on an RTX 4090 on ws6 with both request
and arm orders. All eight snapshot-by-depth conditions reproduced
identical decision hashes across orders. The 1,057- and 954-source
snapshots supply 205 and 193 queries. Every ordered read plan and
response matches the full baseline in all conditions.

| Snapshot | Sources, all / linked | Full / linked source role payload | Packed linked roles | Added bit-mask payload |
| --- | ---: | ---: | ---: | ---: |
| Extended | 1,057 / 140 | 558,450 / 82,284 B | 1,657 B | 1,934 B |
| Confirmation | 954 / 124 | 518,431 / 68,362 B | 1,417 B | 1,575 B |

| Catalog and candidate depth | Packed lookup rows, global → query-local | Packed plan P95, global → query-local, ms |
| --- | ---: | ---: |
| Extended, 128 | 33.94 → 4.83 | 0.286–0.295 → 0.254–0.268 |
| Confirmation, 128 | 34.05 → 6.01 | 0.277–0.497 → 0.244–0.476 |
| Extended connected, 32 | 64.00 → 18.75 | 0.176–0.317 → 0.128–0.208 |
| Confirmation connected, 32 | 64.00 → 20.08 | 0.163–0.164 → 0.123–0.129 |
| Extended connected, 128 | 256.00 → 238.87 | 0.508–0.516 → 0.546–0.556 |
| Confirmation connected, 128 | 248.00 → 248.00 | 0.462–0.472 → 0.510–0.523 |

At depth 128 on the original snapshots, the added filter saves
about 0.02–0.03 ms in planning P95 but adds more bytes than the
entire packed role projection. Connected catalogs benefit only
when the candidate lists leave most links outside the current
query. Deep connected lists make the bit-mask path slower.
Warm local end-to-end differences are small and vary in sign.
The evidence favors the packed global index as the default.
The exact filter remains a comparison for shallow or sparse intersections.

[Aggregate results and replay hashes](results.json)

## Reproduce

Restore both pinned eLife projections and the original open embedding
checkpoint. Repeat with `--reverse` in a CUDA Slurm allocation.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.linked_role_lane_index \
  EXTENDED_PROJECTION benchmarks/elife-extended-role-holdout-v1/manifest.json \
  CONFIRMATION_PROJECTION benchmarks/elife-source-guard-confirmation-v1/manifest.json \
  QWEN_EMBEDDING NEW_RESULTS.json
```
