# Same-inventor retrieval with partial grant rosters

This diagnostic tests whether a search controller can recover inventors
recorded on grants in two technology subclasses under a source-read budget.
It transfers the same-person evidence constraint from scholarly authorship to
a separate source type. The condition is explicit inventorship on two distinct
grants, one per subclass.

## Inputs and evaluation

- The [HUPD January 2016 metadata snapshot](https://huggingface.co/datasets/HUPD/hupd/tree/f570a84b03663180b6034c1f7f4c15864f94385e)
  supplies a grant number, grant date, and the application's primary CPC label.
  We use the first four characters of that label for the subclass. We join only
  on the exact numeric grant ID.
- The [USPTO PatentsView release dated December 31, 2024](https://zenodo.org/records/15058362)
  supplies the grant's ordered inventor roster and disambiguated inventor IDs.
  We preserve source grant IDs as witnesses, check contiguous roster slots,
  and exclude grants that lack a matching roster.
- Development uses grants issued before January 2019. The later grants are held
  out. Within each period, query all pairs of distinct subclasses of the same
  CPC class with at least 12 joined grants on each side. A complete inventory
  ranks inventors by the sum of the reciprocal ranks of their most recent grant
  in each subclass, with rank offset 10. We compare ordered top-five inventor
  IDs and both exact witnesses against an exhaustive implementation.
- Revealed rosters are assigned by a fixed hash of the grant ID at 40%, 70%,
  or 90% availability. Each additional grant roster costs one read. A read
  returns a complete roster. Missingness and queries are constructed from the
  records. This benchmark measures structured retrieval under simulated
  read cost, with no semantic matching, reader latency, or enterprise
  recommendation labels. PatentsView identity resolution may contain errors.

The arms share the visible role metadata. A zero-read query joins the revealed
rosters. The round-robin baseline reads from the two subclass lists in turn.
The static arm estimates the utility of reading each rank from the starting
state. The adaptive arm recomputes priority from the known candidates and the
current top-five boundary after each read. Each of the latter two can also use
certified stopping. The certificate bounds potential rank gains for every
observed individual and anyone absent from both observed rosters. It also
requires the winning grant witnesses to be stable before stopping.

We also build a per-grant role prefilter using 8-, 12-, or
16-bit fingerprints of inventor IDs. A grant whose fingerprint set is
disjoint from the opposite subclass's union cannot support any same-inventor
result. Collisions may retain irrelevant grants. They cannot exclude a
necessary witness. The sketch's byte count is the packed per-grant
fingerprint payload before offsets and runtime union storage. No inventor
identity is returned from the sketch.

The compact exact-ID control models a dense integer code for every inventor
and uses complete rosters to compute the exact top five before
opening the authoritative roster witnesses. With a binding read cap, it
selects the largest confirmable subset of those five and minimizes the
number of distinct newly read grants. Its two-byte-per-incidence payload
excludes offsets and the mapping back to source identities. Treat this as
the stronger fully indexed control, with the same assumption that original
rosters must be read to present missing evidence.

## Results

The exact grant-ID join retained all 19,460 catalog entries and 56,433
inventor appearances. The 14,065 earlier grants produced 415 queries,
of which 83 had eligible inventors in the joined catalog. The 5,395
held-out grants produced 103 queries, 23 with eligible inventors and
54 reference top-five slots.

The table uses the held-out period at 70% roster availability and
at most 12 new reads per query. The fixed and adaptive arms without a sketch
tied. Their certificate also opened all 4,694 unrevealed grant rosters
when required to reproduce the exact ranking and citations on all queries.

| Method | Matched top-five slots | Exact ranked answers and witnesses | New roster reads at cap | Reads for exact outputs |
|---|---:|---:|---:|---:|
| Fixed read order | 34/54 | 94/103 | 1,176 | 4,694 |
| Adaptive read order | 34/54 | 94/103 | 1,176 | 4,694 |
| 8-bit sketch, fixed | 34/54 | 94/103 | 902 | 2,995 |
| 12-bit sketch, fixed | 42/54 | 98/103 | 354 | 666 |
| 16-bit sketch, fixed | 54/54 | 103/103 | 90 | 92 |
| Complete dense-ID index | 54/54 | 103/103 | 26 | 26 |

The 12-bit sketch used 23,873 bytes of packed incidence payload, 75% of
the modeled dense 16-bit payload for this split. The 16-bit sketch and
dense exact postings had nearly identical incidence payloads
(31,844 and 31,846 bytes). The exact control needed fewer reads when
complete coded rosters were affordable. The smaller sketch merits a
direct comparison with partially materialized exact postings at equal
total index size. Counts of exact answers include queries with no
reference inventors in the catalog. Matched slots report the positive
cases separately. All certified sketch runs reproduced all exhaustive
ordered witnesses on both periods. These are roster reads under simulated
availability. End-to-end latency, complete index sizes, and real missingness
remain unmeasured.

## Reproduce

Run with Python 3.10 or newer. PyArrow is used to read the HUPD Feather file.

```sh
study=benchmarks/patent-inventor-replay-v1
curl -fL -o metadata.feather 'https://huggingface.co/datasets/HUPD/hupd/resolve/f570a84b03663180b6034c1f7f4c15864f94385e/hupd_metadata_jan16_2022-02-22.feather'
curl -fL -o inventor-rosters.zip 'https://zenodo.org/records/15058362/files/g_inventor_disambiguated.tsv.zip'
python "$study/build_catalog.py" metadata.feather catalog.jsonl
python "$study/project_rosters.py" catalog.jsonl inventor-rosters.zip rosters.jsonl
python "$study/replay_roles.py" rosters.jsonl results.json
python "$study/exact_index_control.py" rosters.jsonl exact-index.json
```

The scripts verify the source checksums. This directory distributes aggregate
results and scripts. The source metadata, individuals, and grant rosters are
available through the upstream releases linked above.
