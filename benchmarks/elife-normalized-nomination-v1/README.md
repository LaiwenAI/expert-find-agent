# Compare role nomination with a shared-normalization topic baseline

Restore the same 72 eLife requests as the
[live shared-normalization study](../elife-normalized-search-v1).
Search all 322 reports, including the 42 with incomplete declarations.
Only the 280 complete records can enter the exclusion index. The
partial mask makes 144 complete records visible. A source lacking a
visible complete declaration remains eligible. Later exact support
checks use the unchanged publisher projection, which may include
partial positive claims.

Every arm uses the same Qwen3 query encoder, hybrid topical ranking,
one agreement-based role normalization per query, and same-person
role check after the shared relevance reader. A request with an
unresolved role returns before the article reader. All other requests
read exactly eight sources. Compare the first eight topical sources
with two early nomination routes. Both drop only indexed complete
sources that exclude the normalized same-person role pair. One fills
from the first 16 topical sources and one from the entire list.
If the bounded route exhausts compatible sources, it retains
earlier excluded sources to match the reading budget. Assert the
inclusion of all potentially viable original top-eight sources.

Measure ranking, query normalization, relevance verification and
ordered response generation on the same warm GPU. Pairs rotate
execution order. Count tag-backed and wrong-role slots, total issuance,
and per-query tag differences after trimming both ordered lists to
the shorter issued length. The matched count is an evaluation
diagnostic unavailable to either arm at execution time.

## Results

Both tasks in Slurm array 44180 completed on RTX 5090. Non-timing
counts match. Of 36 in-schema queries, 35 enter the common article
reader. The 36 outside-schema queries take no article reads and issue
no recommendation in any arm.

| Complete declarations indexed | Arm | Issued / tag-backed | Sources substituted | Matched count tag delta | Serial P95, ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 144 | Topic | 105 / 58 | 0 | 0 | 84.15–84.35 |
| 144 | Bounded | 108 / 59 | 57 | 0 | 82.49–83.34 |
| 144 | Global | 108 / 59 | 59 | 0 | 82.27–83.10 |
| 280 | Topic | 105 / 58 | 0 | 0 | 84.32–84.86 |
| 280 | Bounded | 110 / 60 | 89 | 0 | 82.37–83.19 |
| 280 | Global | 111 / 60 | 106 | 0 | 82.91–82.98 |

There are no wrong-role slots under the five-field evaluation
contract. The bounded and global paths each gain two tag-backed
slots at full coverage, with no gain once issuance is matched.
Bounded nomination requires fewer substitutions and has similar
milliseconds at this scale. Small timing differences can reflect
source length. Returned titles, source attributions and topic
adequacy do not have independent expert review in this study.
The candidate corpus is small and all requests have authored topic
and duty text. The result isolates one architectural mechanism
against a strong shared-normalization baseline.

[Both GPU runs](results.json)

## Reproduce

Use the checkpoints, manifest and projection from the prior eLife
experiment in an allocated Slurm GPU.

```sh
export PYTHONPATH=src:.
python -m experiments.credit.elife_normalized_nomination PROJECTION \
  benchmarks/elife-role-nomination-v1/manifest.json \
  benchmarks/elife-role-nomination-v1/results.json NEW_RESULTS --order-offset 0
```
