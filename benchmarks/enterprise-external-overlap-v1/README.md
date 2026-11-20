# Screen the external enterprise candidate groups for overlap

Start from the fixed 223 candidate representatives in the
[external sampling frame](../enterprise-external-frame-audit-v1)
and the same 4,301 complete historical workbook records. Verify all
input hashes and reconstruct the request text locally. This is a
deduplication audit. It does not run expert search or inspect
recommendations.

Run two complementary screens. The frozen Qwen3-Embedding-0.6B
encoder embeds the title with the first 640 characters and the title
with the last 640 characters. The semantic score is the maximum
cosine over the four combinations. The local issuer screen compares
normalized names by containment or near-string similarity. For each
screen, separately count candidate groups with any match at 0.85,
0.90 and 0.95 in the other candidates and in the historical workbook.

Build the blind packet from up to 24 pairs at or above 0.90 in each
screen and each scope, sorted by that screen's score. Deduplicate
unordered pairs across screens. Hide the selection channel and
score, and provide separate reviewer copies. If a screen and scope
produce no above-threshold pairs, include eight strongest distinct
nearest-neighbor pairs for exploratory review. This fallback was
added after the first threshold-only pass returned an empty packet. Each reviewer judges
whether the technical needs are equivalent and the organizations
are linked. The thresholds organize review and do not provide
duplicate labels. The workbook is one historical exposure source;
additional registries and organization-level review can still
change which groups can be treated as independent.

The private GPU projection contains only positional text windows
and a frame hash. Organization metadata is used in the local
screen. Private files stay outside Git. Public output includes
counts and the frozen model revision; no candidate IDs, request
text, embeddings, scores, or pair-level matches are included.

## Results

Both tasks of Slurm array `44792` on `ws5` returned the same
counts and private review selections. Neither semantic scope had
a candidate with a score at or above 0.85. The issuer screen found
two candidate groups with an internal nearest match at or above
0.85. Neither screen had an edge at or above 0.90. These are
screen results, without a validated false-negative rate.

The exploratory fallback contributed eight distinct pairs per
screen and scope, with no overlap after deduplication. Both private
copies pass the form validator for all 32 pairs, with zero human
judgments. The 223 representatives remain provisional until the
overlap and independent-cohort checks are resolved.

[Aggregate results](results.json).

## Reproduce

Run preparation and packet creation locally. Run embedding on a
Slurm GPU with `umask 077`, using fresh paths. Repeat the GPU step
with `--order-offset 1` to reverse view order.

```sh
PYTHONPATH=src:. python -m experiments.external_enterprise_overlap prepare \
  SNAPSHOT PRIVATE_WORKBOOK PRIVATE_FRAME PRIVATE_GPU_PROJECTION
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.external_enterprise_overlap gpu \
  PRIVATE_GPU_PROJECTION QWEN_EMBEDDING PRIVATE_SEMANTIC_PAIRS
PYTHONPATH=src:. python -m experiments.external_enterprise_overlap packet \
  SNAPSHOT PRIVATE_WORKBOOK PRIVATE_FRAME PRIVATE_SEMANTIC_PAIRS PRIVATE_PACKET
PYTHONPATH=src:. python -m experiments.check_enterprise_panel_review \
  PRIVATE_PACKET REVIEWER_A REVIEWER_B
```
