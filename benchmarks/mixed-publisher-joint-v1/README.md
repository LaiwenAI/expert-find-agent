# Test first-round expert evidence on a mixed publisher catalog

Keep the 160 PLOS and 51 eLife heading requests from the
[live request benchmark](../live-joint-source-request-v1).
Merge both pinned source snapshots, yielding a 452-article catalog.
Confirm there are no duplicate DOI or case-folded titles and no
shared checksum-valid ORCID across the snapshots. Each publisher's
fixture-local identity tokens receive a namespace, so unrelated
authors cannot be joined through accidental token collisions. This is
a controlled interference test with the same requests and their
existing evidence. The experiment does not assert cross-publisher
identity resolution or generalization to a corpus of web scale.

Rebuild the BM25 and dense index over the merged collection. The
retriever receives both requested topics and can select either
publisher's sources. Compare these fixed candidate pools:

- **Isolated 64** uses the published per-publisher setup.
- **Mixed 64** uses the first 64 global hybrid candidates per condition.
- **Balanced 64** takes 32 from each publisher, preserving the global
  relative order of the selected sources. It receives no indication
  of which publisher supplied a query.
- **Mixed 128** doubles the global candidate depth without increasing
  the number of verification reads.

All source policies open three real local declaration files. Compare
sequential role-gap repair, byline joint nomination, and
joint nomination with all available structured per-person roles.
The latter checks roles in original documents before returning a
response. Candidate enumeration and evidence matching use the same
algorithms throughout. This is a quality diagnostic with a matched
verification budget. It does not time the wider candidate enumeration.

For evaluation, a known match must satisfy both roles and both topic
tags from the query's original publisher. Topics from the other
publisher are unjudged, including otherwise plausible results.
Pool headroom counts requests for which at least one known proof
is fully present before selection, requiring two distinct publications
and one author. The loaded-proof statistic applies the same test after
the three reads, including if a method's chosen proof is different.
Neither statistic adjudicates unlabeled pairs.

Report natural issuance, matched counts on the same issuing requests,
and panel-wide equal-issuance diagnostics. Equal totals select each
method's most confident proposals by weaker citation rank, rank sum
and stable request hash. They may select different requests and are
not a fixed online acceptance rule. Sampling intervals resample
ordered topic pairs and do not address missing labels.

## Results on RTX 5090

Both tasks of Slurm array `44619` on ws5 reproduced identical
aggregates. The per-arm cells give known / issued requests.

| Requests | Pool | Complete known proof in pool | Sequential role gap | Joint byline | Joint personal roles |
| --- | --- | ---: | ---: | ---: | ---: |
| PLOS (160) | Isolated 64 | 104 | 28 / 33 | 49 / 78 | 82 / 131 |
| PLOS | Mixed 64 | 79 | 12 / 15 | 43 / 67 | 68 / 104 |
| PLOS | Balanced 64 | 57 | 6 / 8 | 43 / 60 | 56 / 78 |
| PLOS | Mixed 128 | 115 | 13 / 22 | 45 / 73 | 85 / 143 |
| eLife headings (51) | Isolated 64 | 29 | 0 / 0 | 14 / 16 | 28 / 30 |
| eLife headings | Mixed 64 | 17 | 0 / 0 | 14 / 15 | 17 / 18 |
| eLife headings | Balanced 64 | 15 | 0 / 0 | 12 / 17 | 15 / 24 |
| eLife headings | Mixed 128 | 49 | 0 / 0 | 14 / 16 | 46 / 49 |

Combining catalogs reduces the number of complete proofs in the
64-candidate lists. Widening those lists restores headroom.
Personal-role nomination at width 128 recovers 17 additional known
PLOS requests and 29 additional known eLife requests relative to
width 64. Joint byline gains only two and zero, respectively,
despite the same additional headroom. All responses issued by both
widths are identical for each of the two joint policies. At the
width-64 issuance quota, the known counts remain 68/104 for
personal roles on PLOS and 17/18 on eLife.

The fixed 32-per-publisher quota preserves fewer complete proofs than
the global pool for both panels. It is not a substitute for measuring
candidate interference and the value of depth. In the mixed 128
personal-role arm, only 89 of the 115 PLOS requests with a reachable
proof retain a known proof after the three reads. The final choice
returns 85. These are separate opportunities for selection and
ranking improvement. The wider personal-role arm also returns six
PLOS and one eLife recommendations from the other publisher.
Those recommendations lack aligned topic labels in this benchmark.

## Reproduce

Restore the pinned source snapshots and embedding checkpoint used in
the live request benchmark. Run twice on a Slurm GPU, adding
`--order-offset 1` and using a fresh path for the second output.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.mixed_publisher_joint \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json NEW_RESULTS
```

[Replicated aggregates and paired diagnostics](results.json).
