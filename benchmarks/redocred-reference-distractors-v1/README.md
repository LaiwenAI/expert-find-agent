# Separate competing mentions from threshold-driven attribution rejection

Reuse the exactly four-sentence global window and candidate rectangle
from the [reference coverage audit](../redocred-reference-window-v1).
Use the same frozen Qwen3-Reranker-0.6B and input wording. Calibrate on
63 development cases. Evaluate 62 official test cases and an
additional 237 cases from upstream training titles disjoint from
calibration and official test. The additional cases were used in
the earlier attribution-head study. No models or rules are fitted
on them here. Report them as a separate extension.

## Fixed-candidate comparison

For each claim, start with the same four sentences. If one sentence
mentions both target entities by their canonical names, remove
sentences that name the competing person or work without naming
the corresponding target. The complete target sentence is kept.
Without a target anchor, preserve all four sentences. This rule
never looks at evidence annotations. Compare with the original
shared window and a length control retaining the earliest matching
number of sentences per claim.

Compute a separate cutoff for each arm at the same 5% development
cap on accepting unjudged links. Also apply the original shared-window
cutoff to all arms. As a retrospective analysis, allow each arm
to reach its highest known count without exceeding the original
route's observed unjudged count on that evaluation panel. Tied
scores enter together. Unjudged links are potentially valid,
so these counts characterize review workload.

| Dataset and route | Known accepted | Unjudged accepted | Known accepted at the original cutoff | Known at matched observed unjudged cap |
| --- | ---: | ---: | ---: | ---: |
| Official test, shared | 33 | 4 | 33 | 40 |
| Official test, length control | 28 | 4 | 28 | 43 |
| Official test, candidate scope | 40 | 4 | 30 | 47 |
| Extension, shared | 109 | 30 | 109 | 109 |
| Extension, length control | 102 | 27 | 102 | 109 |
| Extension, candidate scope | 127 | 32 | 111 | 117 |

The retrospective controls use at most four and 30 unjudged links on
the respective panels. Candidate scope preserves all 60 complete
reference windows on the official test and 278 of 283 on the
extension. The length controls retain 49 and 212. Despite trimming
sentence placements from 992 to 888 and from 3,792 to 3,290,
candidate scope does not reduce observed local P95. The test
shared and scoped routes take about 14.6 and 14.9 ms, respectively.
The frozen operating point retains seven additional known links
on test, with a title-clustered interval for the mean per-case
difference of [0.016, 0.234]. At the original cutoff the same input
intervention gives up three known links. This contrast distinguishes
the effect of rerendering the context from the combined effect of
rerendering and recalibration.

## Oracle control on known links with complete reference sets

For each audited relation, give all four candidate claims the same
context. Compare the original window, just the annotated sentences,
and a four-sentence window that retains the annotations and fills
from the closest adjacent sentences. The latter changes the original
window for 47 of 60 official links and 227 of 283 extension links.
The reference selection uses annotations, which makes it a diagnostic
control. Apply the unchanged shared-window cutoff and additionally
show what happens at zero. Report target ordering against the other
person on the same work only when that rival link is unjudged.

| Panel and context | Above zero | Above original cutoff | Known target outscores unjudged rival |
| --- | ---: | ---: | ---: |
| Official test, shared | 54/60 | 17/60 | 13/28 |
| Official test, only references | 50/60 | 11/60 | 26/28 |
| Official test, references plus nearest | 51/60 | 18/60 | 19/28 |
| Extension, shared | 232/283 | 56/283 | 103/177 |
| Extension, only references | 232/283 | 54/283 | 155/177 |
| Extension, references plus nearest | 213/283 | 43/283 | 129/177 |

Reference-only context substantially improves target ordering
against an unjudged rival, while reducing absolute acceptance at the
original cutoff. On the official test, 37 of the 43 shared-window
rejections score above zero. An observed relationship between
ordering and absolute cutoff does not establish that a lower
cutoff would be safe. The reference sentences may also require
unlabeled context. Future qualification verification should track
fixed-context ordering and calibrated issuance separately and
preserve shared contributions.

## Reproduce

Both tasks of Slurm array 45864 on ws3 RTX 5090 reproduce all
non-timing results. Run the command in a Slurm CUDA allocation and
repeat with offset 1. Recover the same input files and checkpoints as
the reference-coverage audit. The recorded sentence indices are
zero-based within the original documents. No document text is
redistributed here. [Metrics and oracle views](results.json).

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.docred.reference_distractor_audit \
  REDOCRED_DIR RE2DOCRED_DIR QWEN_RERANKER NEW_RESULTS --offset 0
```
