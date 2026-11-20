# Size-matched patent role index and request replay

This follow-up uses the same public HUPD/PatentsView grant and query
panels as [the roster replay](README.md). The role
index is materialized as a packed byte array and a four-byte offset
per grant plus one end offset. Each query runs through a loopback
HTTP server. Cached authoritatively available rosters are shared across
arms. Additional source reads execute primary-key point lookups into a
local SQLite database. The reported index sizes include offsets and
bit-packed payloads. The shared grant catalog, inventor ID dictionary,
and raw roster database are common to all arms and are outside that
role index measurement. Process resident memory is not measured.

The 12-bit sketch budget sets the maximum byte count. Under that cap,
partial exact installs complete rosters using 16-bit dense inventor IDs,
leaving other entries empty. The mixed arm first installs 8-bit
fingerprints everywhere, then promotes whole rosters to exact codes
with the remaining space. Both exact admission policies use the same
fixed order, first by the subclass frequency in the query template
workload, then by grant rank. They do not inspect target labels or
the simulated warm roster mask. The complete exact arm uses more space.
The unindexed arm only uses the warm roster cache and source reads.

The endpoint first verifies still-unread witnesses for ranked people it
can assemble from available facts and exact index entries, preferring
candidates with fewer missing witnesses. When no such witness fits the
remaining budget, it reads the next highest-ranked eligible grant. The
sketch safely prunes grants with no fingerprint intersection against
the opposite subclass. Index-derived claims require original roster
witnesses in the emitted response.

## Equal-size results

Held out: 103 subclass-pair queries and 54 positive top-five slots.
The table uses the 70% warm-roster mask, a 12-new-read cap, and no
injected source delay. Latencies are the range of P95 measurements
from three warm rounds, each replayed in a seeded query order.

| Role index | Bytes | Matched slots | Exact results and witnesses | Reads | Client P95, ms |
|---|---:|---:|---:|---:|---:|
| Unindexed | 21,584 | 34/54 | 95/103 | 1,176 | 0.378–0.387 |
| 12-bit sketch | 45,457 | 46/54 | 99/103 | 354 | 0.787–0.826 |
| Sketch plus exact codes | 45,456 | 53/54 | 102/103 | 197 | 0.677–0.718 |
| Partial exact | 45,456 | 54/54 | 103/103 | 26 | 0.655–0.674 |
| Complete exact | 53,430 | 54/54 | 103/103 | 26 | 0.657–0.716 |

Partial exact beat the same-size sketch on matched slots and reads in
all six availability and budget combinations in each period. At the
held-out 70% mask, it already matched the complete exact arm at a
four-read cap. The result is specific to explicit recorded roles,
these subclass pairs, and the workload-informed admission policy.

## Frozen admission and source access cost

Using only the earlier period's query templates to set admission order,
partial exact still retrieved all 54 slots and all 103 ordered answers
with the same witnesses at the 70% mask and 12-read cap. Its reads
increased from 26 to 151, versus 354 for the sketch. At a four-read cap
the earlier-period policy kept all answers and witnesses with 98 reads.
The additional reads show that admission has a material workload effect,
even where ranking and citation checks match.

For the source access sweep, the exhaustive unindexed arm opened all
4,694 masked rosters and returned the same ranked answers and witnesses
as partial exact. The tested slowdown is applied serially before each
new SQLite lookup. Wall time includes client request, server parsing,
index/decode work, roster lookups, and response serialization.

| Added delay per new read | Unindexed, 12 reads max | Unindexed, exhaustive | Partial exact, 12 reads max |
|---|---:|---:|---:|
| 0 ms | 0.369–0.381 ms | 1.242–1.261 ms | 0.644–0.670 ms |
| 0.2 ms | 3.408–3.519 ms | 33.661–34.028 ms | 1.005–1.028 ms |
| 2 ms | 26.058–26.254 ms | 281.069–281.335 ms | 4.507–4.587 ms |

The equal-quality comparison is exhaustive unindexed versus partial
exact. The 12-read unindexed control returned 34/54 slots throughout,
versus 54/54 with partial exact. At zero injected delay, the low-recall
unindexed arm responded faster. At 0.2 and 2 ms per additional read,
partial exact improved both slot recovery and P95 against that arm.
The delay sweep models serial access to slower sources. It does not
measure remote source systems, concurrent load, learned matching, or
enterprise request quality. All timings are from warm, single-client
loopback HTTP on ws5, with sequential policy runs.

## Reproduce

Build `rosters.jsonl` as described in the [roster replay](README.md).
Use Python 3.10+ and an available local loopback port. On a Slurm cluster
run both the HTTP server and client on the allocated compute node.

```sh
study=benchmarks/patent-inventor-replay-v1
python "$study/packed_serving.py" rosters.jsonl scratch serving.json
python "$study/packed_serving.py" rosters.jsonl scratch serving-delay.json --source-delay
python "$study/packed_serving.py" rosters.jsonl scratch serving-transfer.json --development-prior
```
