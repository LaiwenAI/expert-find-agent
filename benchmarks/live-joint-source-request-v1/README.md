# Measure the live local cross-publication request path

Reuse the structured open topic-and-role PLOS requests and held
publisher-heading eLife requests. The request body contains two
topic strings and their controlled roles, with no named expert or
source. Build the sparse and dense indexes offline from pinned
article titles and abstracts. Publisher topic metadata generates
and evaluates requests outside ranking and verification.

For each warm serial request, parse the JSON, encode both topics in
one live Qwen3 0.6B embedding forward, compute BM25 and dense ranks,
and fuse them into a 64-paper list for each condition. Each arm
opens three distinct declaration files and parses their JSON.
Publisher roles and ORCIDs come from those files. Match the two
conditions to the same verified contributor on different sources.
Serialize at most one answer with both citation identifiers. Time
starts at request parsing and ends at response serialization.

| Arm | Source plan |
| --- | --- |
| Topical | One article per condition, then the next topical article |
| Parallel byline | Same first two, then preselect a third using shared byline identities |
| Owner gap | Open the first two, then choose a third conditioned on which contributor roles they support |
| Joint | Join topic pools by indexed ORCID before opening declarations |

Run two verification views. Structured verification uses the rank
pools, canonical roles and original author identifiers. The topic
reader view additionally runs the same frozen Qwen3-Reranker-0.6B
on all six topic–article pairs from each plan in one batch. Require
positive yes-minus-no margin for any cited article assigned to
that topic. The fixed cutoff is diagnostic and is never tuned to
either held-out result. The topic reader does not verify author
identity or replace the role declarations.

Report known outputs supported by both publisher topic labels and
issued outputs whose citations lack at least one of those labels.
An unjudged citation may be relevant, and a reader output is not a
human judgment. Topic-pair intervals are descriptive. Time includes
actual warmed local files and model inference. It excludes network,
queueing, service concurrency, cold startup, creating the index and
acquiring publisher XMLs. The structured query supplies the role
labels and condition boundaries.

## Live results

Both tasks of Slurm array `44460` ran on RTX 5090 on ws5.
All non-timing outcomes reproduced. Counts distinguish known
results and topic-unresolved issued results. P95 is the warm local
request latency across both execution orders.

| Panel | Plan | Structured known / unresolved | Structured P95 (ms) | With topic reader known / unresolved | With topic reader P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| PLOS, 160 | Topical | 1 / 0 | 9.68–9.76 | 1 / 0 | 58.03–58.11 |
| | Parallel byline | 28 / 7 | 9.80–10.17 | 2 / 0 | 58.04–58.08 |
| | Owner gap | 28 / 5 | 9.79–9.85 | 2 / 0 | 58.10–58.20 |
| | Joint | 49 / 29 | 9.79–10.10 | 5 / 0 | 41.40–41.46 |
| eLife headings, 51 | Topical | 0 / 0 | 10.04–10.50 | 0 / 0 | 35.11–35.16 |
| | Parallel byline | 0 / 0 | 10.15–10.89 | 0 / 0 | 35.16–35.18 |
| | Owner gap | 0 / 0 | 10.21–10.43 | 0 / 0 | 35.22–35.29 |
| | Joint | 14 / 2 | 10.23–10.61 | 0 / 0 | 33.22–33.44 |

Joint nomination preserves an increase in known results at a
comparable local structured request cost. It also issues more results
overall. This is a coverage comparison at matched source reads,
with no claim of higher quality at equal issuance. The verifier's
zero cutoff sharply reduces known returns and increases latency.
The model's token workload may differ by plan despite an equal
number of topic–document checks. The 0.6B reader is therefore
retained as a baseline for topical calibration.

## Query execution stability

The earlier cached eLife evaluation found 26 known answers. A paired
encoding audit on the same request handler and static document
index reproduces that count when supplying the cached query-bank
encodings from a single full-panel batch. Encoding each topic
alone or encoding both topics per request produces 14.
The per-request pair changes source sets in 71 of 102 rankings,
the selected reading plan in 16 of 51 requests and the serialized
answer in 13. Maximum absolute vector drift is 0.00402 against
the cached panel. This isolates query execution layout as an
explanation for the earlier result gap under fixed downstream rules.
Both audit tasks in Slurm array `44466` reproduce these counts.

A second audit freezes two input slots of 64 tokens, with absolute
positions or positions relative to the attention mask. On these
inputs, each topic vector is bit-identical to its same-width
two-copy reference when the other slot changes. Both routes still
return 14 known matches. Absolute positions issue 15 total
responses and relative positions issue 16. Local P95 ranges
from 9.82 to 10.11 ms. This fixes neighbor dependence for
the tested layout without restoring the cached panel advantage.
Both tasks of array `44474` reproduce the non-timing aggregates.
The width applies to short structured topics only.

[Live request aggregates](results.json) · [Ranking audit](ranking-stability.json) · [Fixed-width audit](fixed-positions.json)

## Reproduce

Restore both publisher snapshots and the verified Qwen3 embedding
and reranker checkpoints. Run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.live_joint_source_request \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```

Repeat with `--order-offset 1`.


The query layout audits use the same eLife XML, manifest and
embedding checkpoint:

```sh
PYTHONPATH=src:. python -m experiments.credit.live_joint_rank_stability   ELIFE_XML ELIFE_MANIFEST QWEN_EMBEDDING NEW_RESULTS
PYTHONPATH=src:. python -m experiments.credit.live_joint_fixed_positions   ELIFE_XML ELIFE_MANIFEST QWEN_EMBEDDING NEW_RESULTS
```
