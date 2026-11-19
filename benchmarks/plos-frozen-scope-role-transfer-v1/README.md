# Test legacy scope and activity verification on modern CRediT records

Freeze the four main verification arms, the Qwen3-Reranker-0.6B
revision and the contribution-section thresholds from the
[legacy article experiment](../plos-factored-article-role-v1).
This transfer uses modern PLOS contributor-role fields rather than
legacy prose notes with initials. The eligible publications come
from the [pinned external publisher inventory](../plos-credit-external-v1).
Exclude any DOI or normalized byline name appearing in the legacy
captures. Verify each XML checksum, declared license and contributor
roles against the inventory before constructing cases.

Fix one case per eligible article and requested role, with a target
author and three windows. The first is the target person's declared
role. The second shows that same person's different role from a
fixed confusable activity group. The third shows another person's
declaration for the requested role. In the third case the target
person also holds the requested role, but the offered citation
credits a different person and cannot be attached to the target.

The joint verifier sees a rendered name and one declared role per
window. The modern scope gate compares the target author ordinal
with the containing JATS contributor element, using the document's
structured association rather than matching names in a rendered
string. The scoped joint arm keeps the joint prompt for the two
passing windows. Scoped role drops the requested author from the
prompt and still reads the full rendered window. Scoped head reads
the exact role field after the same owner check. All variants
receive the same original evidence options and use the original
legacy thresholds with no modern calibration.

Evaluate the same cases with two existing, authored query banks.
Straightforward wording paraphrases the requested role; boundary
wording often mentions a different contributor or a nearby role.
Choose one expression per article and requested role by stable hash.
A clean result accepts the correct declaration and rejects both
foils. Report separate counts for wrong roles and wrong evidence
ownership, and intervals clustered by normalized author components.
These are controlled attachment decisions for already available
windows. Timing covers filtering, prompt preparation and a warmed
model forward, excluding retrieval, index construction and queues.

## Results

Slurm jobs 43106 and 43110 ran on one RTX 5090 on ws5 and reproduced
all frozen outcome counts. Of the 24 source articles, 22 supply
both contrast types, with 127 cases across all 14 roles. The
normalized bylines form 21 clusters. This is a same-publisher,
different-format transfer.

| Wording | Verifier | Correct accepted | Wrong role accepted | Wrong owner accepted | Clean cases | Model pairs | Verifier P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Straightforward | Joint | 106 | 36 | 1 | 70 | 381 | 8.86 |
| | Scoped joint | 106 | 37 | 0 | 69 | 254 | 8.83 |
| | Scoped role | 122 | 63 | 0 | 61 | 254 | 8.86 |
| | Scoped head | 33 | 0 | 0 | 33 | 254 | 8.72 |
| Boundary | Joint | 74 | 26 | 0 | 58 | 381 | 8.89 |
| | Scoped joint | 73 | 27 | 0 | 57 | 254 | 8.89 |
| | Scoped role | 121 | 81 | 0 | 46 | 254 | 8.77 |
| | Scoped head | 29 | 0 | 0 | 29 | 254 | 8.81 |

The structured owner gate removes one third of model pairs, by
construction of this contrast, with small measured P95 changes.
The head-only verifier accepts no foil at the legacy threshold,
but loses 37 and 29 clean cases relative to the joint verifier.
The author-cluster 95% intervals for those changes are −37.58 to
−18.96 and −32.19 to −12.90 percentage points.

To distinguish calibration from semantic ranking, compare the
correct and wrong roles of the same candidate before thresholding.
Scoped head ranks the correct role higher on 126/127 straightforward
and 96/127 boundary cases. Three boundary pairs tie. A retrospective
sweep of one global threshold on the evaluation scores could retain
up to 62 and 38 correct declarations with no accepted same-author
foils, versus 33 and 29 with the frozen threshold. These are
label-informed diagnostics, not deployable thresholds. The boundary
phrases have both lower role separation and limited headroom from a
threshold change. Independent calibration may help, while a
near-role contrast model still needs evaluation on separately
authored natural requests.

[Aggregate results](results.json)

## Reproduce

Use the pinned legacy and modern captures referenced above and run
on a Slurm CUDA compute node. The modern snapshot is the original XML
directory from the external capture.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.frozen_scope_role_transfer \
    OLD_PUBLISHER_SNAPSHOT LEGACY_TRANSFER_SNAPSHOT MODERN_PUBLISHER_SNAPSHOT \
    benchmarks/plos-credit-external-v1 benchmarks/plos-credit-transfer-v1/phrases.json \
    benchmarks/plos-factored-article-role-v1/results.json RERANKER_CHECKPOINT NEW_RESULTS.json
```
