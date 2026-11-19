# Recheck competing work assignments without forcing exclusive roles

The [work and owner contrast](../redocred-work-owner-contrast-v1)
found that subtracting the score for the same person's other
work can retain more disjoint links, but can suppress valid
shared contributions. This experiment uses that comparison to
select a full-document recheck. The initial evidence window
contains at most four shared sentences.

Preserve an initial high-scoring link. Consider another link
for full-context verification only when its compact score lies
between a lower development cutoff and the initial cutoff.
The scope route additionally requires the same person's score
on the competing work to be at least as high as the link's
score. This detects a plausible relation that would lose a
relative comparison. Run a single full-context batch for that
document if any such link is eligible. A link is added only
if its own full-context score exceeds a nonnegative threshold.
Independent assertions can coexist.

The score-only recheck control uses only the compact-score band
to nominate links for full context. Its calibration dispatch
count may not exceed the scope route's. Fit the high compact
cutoff and full-context cutoff on development cases by maximizing
accepted known links under the same 5% unjudged-acceptance
ceiling. Select fewer rechecks and then fewer accepted unjudged
links on ties. The lower cutoff comes from the compact absolute
20% calibration setting. Also report compact absolute, compact
work margin and full absolute with the corresponding 5% cutoffs.

Use the previous development cases for calibration. Test on
titles in the upstream training release that are disjoint from
its development and test documents. These titles are held out
from this policy fit. The benchmark gives no guarantee about
foundation model pretraining exposure. Select at most 120 cases
in each original-label class by fixed hash order, with no model
output used in selection. Group the reported results by the
updated panel's two-link and shared-link structure. Use both
original and extended labels to construct and score the cases,
as in the earlier contrast.

Execute both evidence arms on the GPU for every calibration and
test case. Replay the fitted dispatch on those paired batches.
The serial estimate adds measured compact and full timings for
the cases selected for recheck. The replay charges one full
batch for all four links, and accepts only the eligible links
that pass the full-context cutoff. It excludes queue and service
overhead and does not time a deployed cascade. Unjudged links
can be missing positive annotations.

## Results

Slurm job 43354 completed on an RTX 5090 on ws5. The test
contains 240 cases from 189 documents, with 636 known and 324
unjudged links. There are 120 two-link cases and 120 shared-link
cases under the updated annotations.

| Frozen route | Accepted known | Accepted unjudged | All known retained in shared cases / 120 | Full batches | Serial P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact absolute | 245 | 24 | 18 | 0 | 71.2 ms |
| Compact work margin | 268 | 10 | 0 | 0 | 71.2 ms |
| Full absolute | 179 | 17 | 8 | 240 | 118.2 ms |
| Scope recheck | 245 | 24 | 18 | 0 | 71.2 ms |
| Score-only recheck | 245 | 24 | 18 | 0 | 71.2 ms |

The scope-recheck grid selects no full-context dispatch. No
evaluated active policy retains more known development links
than compact absolute under the 5% unjudged-acceptance
ceiling and the chosen score band. The score-only comparator
inherits the resulting zero-call calibration budget. The two
fallback rows therefore reproduce the compact baseline. This
experiment rejects adoption of the evaluated full-context
fallback at this operating point and leaves other rechecks
unresolved.

The work margin replicates the separation observed on the
earlier test. It accepts 139/240 known disjoint links versus
63/240 for the compact absolute gate, but 129/396 known shared
links versus 182/396. Its overall gain is 23 known links.
The paired difference per four claim slots is +2.40 points,
with a descriptive document-bootstrap interval of −2.01 to
+6.76. On the independent panel the compact absolute gate
accepts 24/324 unjudged links, above the 5% development
ceiling. This cautions against transferring an operating
threshold across document cohorts without checking coverage
and risk. Unjudged returns do not measure confirmed errors.
[Aggregate results](results.json)

## Reproduce

Use all three original and extended splits from the pinned
[original release](https://github.com/tonytan48/Re-DocRED/tree/ccfb54f5ddf5836027c87badda10f6dfc56efaac/data)
and [extended release](https://github.com/klassessg/re2-docred/tree/006412ef599a3fd6039b3746cc2d1e22fc600d62/dataset),
and the same frozen open 4B reader as the work and owner
contrast. The runner verifies each input used in calibration,
exclusion or evaluation. Run within a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.scope_triggered_recheck \
  REDOCRED_DIR RE2DOCRED_DIR QWEN4B NEW_RESULTS
```
