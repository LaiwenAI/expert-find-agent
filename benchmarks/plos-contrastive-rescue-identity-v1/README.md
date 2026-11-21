# Verify recovered source identity independently of author linkage

The [previous PLOS replay](../plos-recoverable-source-calibration-v1)
showed that an author can satisfy the requested roles on a substitute
paper after the intended source disappears. Calibration suppressed
that error at the cost of some recoverable proofs. This follow-up
tests whether the text of each clue identifies its proposed source
before the same eLife-trained pair scorer can accept it.

## Comparison

Keep the pinned 414-paper PLOS catalog, two disjoint 12-group
calibration and test partitions, same-role author postings, learned
role decoder, and source scorer. Leave an already-issued shallow
result untouched. Apply the following checks only to proposed
fallback pairs.

| Guard | Pair is eligible when both papers pass |
| --- | --- |
| Unfiltered | Existing author and exact role membership is sufficient to reach the scorer |
| Top-three and cosine | Each paper ranks in the top three for its clue and has greater cosine similarity to that clue than to the other clue |
| Distinctive-term floor | Each paper contains at least half of the weighted distinctive clue terms |
| Unique distinctive coverage | Each paper passes the term floor and strictly exceeds the second-best remaining paper on distinctive-term coverage |

The distinctive terms are the terms in one clue absent from the
other clue. Weight them by the existing index's document-frequency
weights. Compute coverage from the frozen indexed text. Compare
against the entire available catalog, before considering author
or role membership. A tied top coverage fails uniqueness. The
top-three and cosine check adapts the criterion from the
[eLife held-source guard](../elife-source-guard-confirmation-v1)
to fallback candidates only.

Compare the frozen eLife cutoffs and re-calibrate each arm on
the PLOS calibration groups with both natural and artificially
demoted-source requests. Evaluate exact proof, substitute issuance
under withdrawal, and the eligible candidate menu before scoring.
Use the same withdrawal and demotion procedures. The rank guard
acts on the intervened rank, so it rejects the designated source
if moved to position seven. The lexical checks use the indexed
text and the same catalog in the original and demoted views.
The demotion experiment exposes reachable misses and is separate
from intact natural search. The unfiltered arm must reproduce
the archived natural and demotion outputs.

The PLOS test reuses previously inspected groups, and the selected
half-coverage threshold is exploratory. This study counts candidate
choices and measures exact publisher-declared evidence. It does
not estimate deployment risk or client latency.

## Results

Both runs in Slurm array 47323 on ws6 RTX 4090s reproduce every
count and cutoff with reversed group order. The table shows the
author-role route on the held PLOS groups, using frozen eLife
thresholds. Keyword intake has 72 natural requests with both
sources present, 144 source-withdrawal requests, and 132
demotion requests covering 22 source-side interventions.

| Keyword fallback guard | Correct natural proofs / 72 | Wrong proofs on withdrawal / 144 | Correct proofs on demotion / 132 | Wrong proofs on demotion / 132 |
| --- | ---: | ---: | ---: | ---: |
| Unfiltered | 10 | 2 | 17 | 3 |
| Top-three and cosine | 9 | 0 | 0 | 0 |
| Distinctive-term floor | 9 | 0 | 2 | 0 |
| Unique distinctive coverage | 9 | 0 | 2 | 0 |

Before scoring the natural keyword fallback, author-role lookup
exposes one exact option and six wrong options across all source
availability conditions. Every identity guard removes all seven.
The two wrong withdrawal outputs occur in one paper group. On
demotion, the term floor keeps two of 17 exact options and
removes six wrong options before the model scorer. Requiring
uniqueness changes no results relative to the floor. Calibrating
again in PLOS leaves the two lexical recovery outputs; the
calibrated unfiltered arm recovers none. The [previous soft
lexical scorer](../plos-recoverable-source-calibration-v1) retains
ten recoveries at zero observed keyword errors under the same
intervention and also retains nine natural correct proofs.

On excerpt intake, all three hard guards reject every author-role
fallback option, including 18 exact options after demotion.
With the guards there are still two wrong author-role outputs
under excerpt demotion. Both are inherited from shallow
search; the fallback has no eligible options. Global fallback
under the frozen eLife thresholds previously issued four
substitutes on source withdrawal in each intake. The guards
remove them and also remove nearly all recoverable candidates.
Publisher-declared proof counts under withdrawal assess the
designated source, and are separate from expert recommendation
accuracy.

These results locate two limitations. The natural author-role
fallback presents almost no exact new options on this panel.
A stricter acceptance rule cannot expand that menu. Also,
checking fallback alone leaves errors already emitted by shallow
search. Preserve the soft scorer as an experimental comparator
for deliberately recoverable misses. The hard lexical rules have
no observed natural quality gain and cannot replace it as the
default for source recovery.

[Aggregate results and replication digests](results.json)

## Reproduce

Use the same pinned inputs as
[the archived calibration replay](../plos-recoverable-source-calibration-v1#reproduce).
Add that replay's archived result before the new output argument.
The eLife pair file and held panel may need to be restored at their
original storage paths. The runner verifies their content hashes.
Use a fresh output path and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.contrastive_rescue_identity \
  ELIFE_CONFIRMATION ELIFE_EXTENDED ELIFE_FROZEN_GROUPS ELIFE_HELD_PANEL \
  ROLE_DECODER_RESULTS ELIFE_DIRECTED_RESULTS BASE_PLOS_MANIFEST BASE_PLOS_PROJECTION \
  PROSPECTIVE_PLOS_MANIFEST PROSPECTIVE_PLOS_XML PROSPECTIVE_PLOS_RESULTS \
  NISO_EXAMPLES ROLE_TRAINING QWEN_EMBEDDING \
  benchmarks/plos-recoverable-source-calibration-v1/results.json NEW_RESULTS
```
