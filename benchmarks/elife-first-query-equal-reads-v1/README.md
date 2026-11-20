# Time first-query role nomination at equal and adaptive read counts

Replay the 60 test queries from the
[eLife factorial study](../elife-first-query-factorial-v1), with its
frozen relevance cutoff, 280 complete indexed declarations, and the
same BGE-small hybrid ranker and MiniLM L12 verifier. There are four
first-stage reads, an eight-source local pool defined by the ranker,
and up to two second-stage reads. The adaptive routes run local rescue
when fewer than five people have passing evidence. Record skipped
reads separately when the five slots are already filled or the local
pool contains too few unseen role-compatible sources.

The fixed routes verify two additional sources on every query.
The fixed local routes prefer the same compatible sources selected
by the adaptive policy. If they run out, they read other unseen
members of the original pool, including articles already known to
lack the requested same-person role pair. This explicitly equalizes
verification work and exposes what changes when a local pool is
exhausted. An additional fixed route fills from the global compatible
directory after early role nomination. All five routes use the same
final ranking and deduplication.

Use a live verifier for both stages, with no cross-arm score cache.
Rotate arm execution order by request and repeat with a different
starting offset. Each panel warms all arms on the same first request
before timing. Request wall time includes query encoding, hybrid
ranking, candidate selection, first-stage and rescue verification,
and assembling the source-backed people. It excludes setup and
service queueing. Timing uses Apple M4 and one ONNX intra-op and
inter-op thread. The endpoint is publisher tag-backed evidence and
off-label citations remain unresolved.

## Results

Both orderings give the same counts. Ranges are the serial P95 observed
across the two runs. Keyword rows have 36 requests and 180 fixed
recommendation slots.

| Keyword route | Tag-backed slots | Off-label slots | Reads | P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Topic, adaptive local | 49 | 21 | 195 | 320.7–335.0 |
| Person, adaptive local | 51 | 22 | 166 | 268.0–270.4 |
| Topic, fixed local | 49 | 21 | 216 | 350.5–353.5 |
| Person, fixed local | 51 | 22 | 216 | 330.0–330.9 |
| Person, fixed global | 52 | 23 | 216 | 344.4–344.5 |

The 29-read adaptive difference comprises 27 additional unfilled
read slots from local compatibility exhaustion and two additional
skipped reads when five people already have passing evidence.
Equalizing reads keeps the two-slot tag-backed difference and the
one-slot off-label difference. Globally allocating the remaining two
reads adds one tag-backed slot and one off-label slot against fixed
local; it also increases local P95. Padded reads equalize verifier
calls and article counts, with article length and inference work still
able to differ. The speed comparison pertains to this service-free
warm local replay.

On headings, all five arms return five tag-backed and two off-label
slots. At equal reads, the person-local route is slower than the
topic-local route (365.6–381.6 vs 344.2–346.5 ms). The earlier
cached factorial reported zero heading off-label slots. The live
two-stage scoring schedule retains the same tag-backed counts and
gives two off-label slots in both execution orders. Exact contextual
equivalence of cache and live scoring remains unverified.

[Aggregates for both execution orders](results.json)

## Reproduce

Use the projection and checkpoint files from the factorial benchmark.
The runner verifies their hashes against its public result. Run each
order on an otherwise idle machine before comparing serial P95.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.credit.role_first_query_equal_reads PROJECTION \
  BGE_MODEL_DIRECTORY FLASHRANK_CACHE_DIRECTORY \
  benchmarks/elife-first-query-factorial-v1/results.json NEW_RESULTS \
  --order-offset 0
```

Repeat with `--order-offset 2` and a fresh output path.
