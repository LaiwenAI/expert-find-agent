# Bind prior experience to its task or work content

Use the public anonymized [GOJA dataset](https://github.com/KruegerETRF/GOJA/tree/06c9887cceafdacec319ea222b16a4419870eca6)
from [Krüger et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1375/).
GOJA provides human-annotated spans and relations in 500 German
job ads. This diagnostic isolates whether an existing experience
anchor can bind to the annotated scope when another plausible
candidate occurs nearby. The dataset is licensed CC BY-NC 4.0.
This repository includes the code and aggregate results. Obtain the
anonymized data from the upstream source.

## Fixed task

- Read the pinned XMI annotations using UTF-16 offsets. Group and
  deduplicate identical normalized ads. An experience span and a
  process or work-content span form a gold pair when GOJA connects
  them with `Nullrelation` or `Arbeitsmittel`. Treat the edge as
  undirected for binding.
- Keep ads with at least two linked experience anchors. Collect the
  target spans already linked to an experience anchor in the same ad,
  within 180 characters of the current anchor. Keep the five nearest
  by distance with source order as the tie-breaker. Select cases with
  one annotated target and at least one distractor in that menu.
  Exclude distractors that overlap or touch any annotated target for
  the anchor. Menus have two to five candidates.
- Hash normalized ad text to split by document. The first six hash
  buckets train the small model and the remaining four are held out.
  The test contains 150 anchors from 65 ads. The fit uses 298 anchors
  from 117 ads. Confidence intervals resample ads, including multiple
  anchors from the same ad as one unit.

Compare nearest by character gap, nearest within the same line
when possible, and a regularized linear ranker. The ranker uses
log character gap, whether both spans share a line, whether the
candidate follows the anchor, and whether the target is an annotated
process. Weight each anchor equally, with half its weight on the
positive and half split over its distractors.

## Structural results

| Route | Correct target / 150 |
| --- | ---: |
| Nearest by gap | 134 |
| Prefer same line, then nearest | 143 |
| Learned structure | 147 |
| Learned structure applied after flattening line breaks | 145 |
| Learned structure fitted and tested on flattened text | 145 |

The learned structural ranker gains 13 correct bindings over nearest.
The paired difference is +8.7 percentage points, with a grouped 95%
interval from +4.5 to +13.3. Flattening line breaks while keeping
spans, candidate order, and distances unchanged loses two correct
bindings. Refitting on flattened training text gives the same count.
The paired interval for the frozen ranker on flattened versus original
layout is −3.4 to 0.0 percentage points.

The strongest geometry-only route leaves three errors in this
filtered panel. Use it as a reference when testing whether a live
semantic verifier helps on harder menus or less reliable document
layouts. This is supervised scope binding with annotated mentions
and an available target. It does not measure mention extraction,
qualification necessity, expert matching, or whole-request latency.
The split screens exact ad duplicates. Near duplicates and shared
posting templates may remain.

Slurm job 46710 reproduced the CPU results in both traversal orders.
The layout intervention used the same pinned archive and split.
[Aggregate results](results.json)

## Reproduce

Download `data/goja_redacted_v1.zip` from the pinned dataset revision.
It has SHA-256
`a1b1408656283268470e8103b676229dd8e7e692fd361a1ed463460c32ce950c`.
Run on a Slurm compute allocation. Repeat the first command with
`--reverse`.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.goja_experience_binding GOJA_ZIP NEW_RESULTS
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.goja_layout_ablation GOJA_ZIP NEW_LAYOUT_RESULTS
```
