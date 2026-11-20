# Recover same-author declarations from selected source spans

Test the fallback after the role index identifies a publisher note but
does not have its author postings in memory. The
[independent PLOS capture](../plos-independent-cascade-v1/capture.json)
contains older publisher contribution notes formatted as task heading,
colon, and author initials. The study supplies the target paper and two
role labels and compares all ten pairs of the five legacy role labels
per admitted note.

For each distinct task heading, the offline index stores its byte
range in a whitespace-normalized copy of the note and the names
eligible for unambiguous initials resolution. The compiler checks its
recognized clauses against the existing publisher-note parser. A note
allows definitive absence only when every clause is recognized, every
initial resolves, and headings are unique. With an incomplete note,
known clauses can support a same-author claim; absent support stays
unresolved. Duplicate headings and ambiguous initials are not forced
into ownership postings. Source XML and license checks run offline.

All methods use the same cache mask and complete-heading metadata.
Without enough cached evidence, the baseline reads and parses the
entire normalized note. A second method reads only the unresolved
roles' ranges. A third reads one contiguous range covering both
required spans. Each returns named-author support, no declared
match within that paper, or unresolved. Unknown headings are carried
by the unresolved branch. The comparison bypasses a note read if
cached postings suffice or a complete heading index excludes a role.
The resident route is an in-memory ceiling. Byte totals count the
payload handed to the parser, rather than physical storage traffic.

## Results

Job `43638` ran on a Slurm CPU allocation on ws5. Of 64 pinned
records, 58 had usable licensed notes and 27 passed the completeness
check. In every cache condition, all 580 queries returned the same
outcomes across all three strategies. The 428 supported requests
have at least one author linked to both recognized clauses, 38 have
no declared match in a complete note, and 114 remain unresolved.

| Owner postings cached | Whole note bytes | Separate spans bytes | Coalesced span bytes | P95 in same order |
| --- | ---: | ---: | ---: | --- |
| None | 109,646 | 40,559 | 59,322 | 11.40 / 6.30 / 6.32 µs |
| Hashed 50% target | 76,521 | 19,343 | 23,513 | 11.56 / 5.84 / 6.03 µs |
| Resident | 0 | 0 | 0 | 0.81 / 0.73 / 0.81 µs |

At zero owner-posting cache, coalescing cuts delivered bytes by 45.9%
and retains the same 545 note opens as whole-note reading. Separate
spans cut bytes by 63.0%, with a higher number of application reads
(1,042 versus 545). The intentionally warm local storage was exercised
five times per case with interleaved method order. Preparing all
records, including XML checks and note materialization, took 108 ms.
The local timing captures extent lookup, reads, and parsing for very
short notes. It does not measure cold storage, source retrieval, query
interpretation, or opaque prose. Matching the complete-note parser
establishes execution equivalence with the scoped structure. It is not
an independent assessment of contribution claims in the broader article.

[Aggregate results](results.json)

## Reproduce

Restore the publisher JATS files with the
[fixed capture receipt](../plos-independent-cascade-v1/capture.json).
Use one CPU on a Slurm compute node.

```sh
PYTHONPATH=src:. python -m experiments.credit.clause_fallback SNAPSHOT NEW_RESULTS.json
```
