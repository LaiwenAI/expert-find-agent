# Publisher proof before semantic reranking

This live transfer test applies the [PLOS coverage-aware screening
design](../plos-topic-role-holdout-v1/README.md) to the pinned
[322-paper eLife projection](../elife-role-nomination-v1/README.md).
It tests two- and three-role conjunctions tied to a single source and
the same author. For each topic with at least four fully declared
articles, use the rarest and most frequent nontrivial covered role
conjunctions, with deterministic tie-breaking. Each role is rendered
from the existing paraphrase bank into an explicitly typed input line.
The topic is supplied separately to all arms.
Each three-role request also has a paired control with an extra
mandatory free-text condition unsupported by this evidence path. The
conditions remain in the input packet, so the route abstains on these
controls before inference. This is a routing boundary check with known
input typing. It does not calibrate detection of hard conditions in
unstructured prose or compound clauses.

Every arm shares the frozen Qwen3-4B-Instruct-2507 menu normalizer, the
Qwen3-Embedding-0.6B hybrid topical ranker, the Qwen3-Reranker-0.6B
semantic reader, its frozen topical cutoff, the full positive
publisher declaration check, and a five-expert output cap. For the
screening arms, source-level absence of a same-author conjunction can
exclude a source only if its complete declaration is visible to the
index. An incomplete or unindexed declaration can still provide
positive support at the downstream check. The partial mask exposes
144 complete declarations and the full mask 280. The actual topical
verification budget is four or eight sources.

| Arm | Behavior |
| --- | --- |
| Late proof | Rerank the initial topical set, then check the author's declaration. |
| Compact | Remove provably ineligible sources from that exact same set before reranking. No backfill. |
| One-wave refill | After screening, fill the freed budget with later compatible sources before one reranker pass. |
| Gap refill | Verify the compact set first, then use remaining budget only if fewer than five experts are confirmed. Keep existing recommendations and citations in their original order. |

Under stable candidate-level scores, compacting a fixed initial set
cannot change the final recommendations or citations. Actual batching
can shift scores near the topical threshold or disturb their relative
order, so the experiment checks exact ordered output instead of
assuming equivalence. The two refill routes can change coverage and
citation selection even under stable scores. Count topic-label hits
among author-role-supported outputs. Also trim each arm and its
reference to their per-query shared issuance count and report the
paired topic-group interval for that comparison. Issuance trimming
is an evaluation diagnostic. Topical tags are incomplete and provide
no independent judgment of relative expert fit.

P95 includes local warm packet construction, query interpretation,
query encoding, hybrid retrieval, one or two live verification
batches and final recommendation construction. Model startup,
article encoding, index construction, network, and serving queues
are outside the timed interval. Two Slurm runs rotate arm ordering.

## Results

Both tasks of Slurm array `44230` completed on an RTX 5090 on ws5,
with identical non-timing aggregates. The panel covers 18 topic groups.
Each group has two two-role and two three-role requests, plus the two
paired open-condition controls. The shared normalizer exactly maps
32 of 36 two-role and 33 of 36 three-role requests. The other seven
abstain. All open-condition controls abstain before model inference.
Issued recommendations retain the requested publisher roles in these
views. Topic relevance is still measured by publisher tags.

The table shows the three-role cohort at the full mask. Counts are per
run over 36 requests. The latency range spans both runs.

| Cap | Route | Reader pairs / calls | Issued / tag-backed | Matched-issuance tag delta | Identical results to late proof | P95 (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | Late proof | 132 / 33 | 54 / 36 | 0 | 36 | 86.04–86.08 |
| 4 | Compact | 63 / 29 | 54 / 35 | −1 | 34 | 80.47–80.74 |
| 4 | One-wave refill | 132 / 33 | 72 / 38 | −4 | 20 | 87.25–87.79 |
| 4 | Gap refill | 130 / 58 | 73 / 41 | −1 | 21 | 89.21–89.75 |
| 8 | Late proof | 264 / 33 | 75 / 43 | 0 | 36 | 106.79–106.97 |
| 8 | Compact | 134 / 33 | 77 / 44 | 0 | 33 | 95.40–95.48 |
| 8 | One-wave refill | 264 / 33 | 83 / 45 | 0 | 30 | 105.13–105.46 |
| 8 | Gap refill | 250 / 58 | 84 / 45 | 0 | 28 | 105.63–105.89 |

At cap eight, compacting the three-role set removes 49.2% of pair
scores with full coverage. At partial coverage it removes 27.3%
(264 to 192 pairs), and exact outputs agree on 34 of 36 requests.
The two-role cap-eight full-mask replay similarly cuts 256 to 168
pairs and moves P95 from 85.76–85.77 to 74.11–74.56 ms, with exact
output agreement on 33 of 36 requests. Changing the reader's batch
shape is therefore material even when the candidate removals preserve
eligibility under fixed scores. This run does not attribute individual
mismatches to the acceptance threshold or to score order.

At cap four the three-role gap route adds five tag-backed slots at
full coverage, with one fewer tag hit after matching issuance. At cap
eight its matched amount is unchanged. The one-wave route likewise
has zero or negative matched tag deltas at full coverage. Across all
masks and caps, the paired intervals do not establish a positive
matched-issuance gain. The deferred wave makes more reader calls
than the one-wave route in this benchmark. The latency and coverage
results motivate a stronger fixed prescreening baseline and a
shape-stability test, with independent expert judgments needed to
validate any replacement benefit.

[Both GPU aggregates](results.json)

## Reproduce

Use the pinned projection, manifest and reference from the eLife
role-nomination benchmark. Run in a CUDA allocation with the same
local checkpoint layout.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.elife_proof_before_rerank \
  PROJECTION benchmarks/elife-role-nomination-v1/manifest.json \
  benchmarks/elife-role-nomination-v1/results.json NEW_RESULT_JSON --order-offset 0
```
