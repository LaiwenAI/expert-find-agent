# Plan nominations around sources under the same evidence budget

The [nomination comparison](../trec-supported-owner-nomination-v1)
found that optimizing the availability of evidence for individual
nominees increased the read oracle ceiling while reducing observed
supported recommendations. Test whether selecting the nominee list
from a feasible source plan closes that gap.

Reuse the original 49 TREC Enterprise topics. Remove the held
topic's judgments before fitting any model or building the
historical graph. Each training topic also masks its own judgments
when constructing features. Use the same historical source menus
of up to 16 documents per owner. The original first 20 people or
the original first 60 people define the two candidate universes.
All routes nominate 20 and share the original reuse route's
realized read and check caps per held topic. These matched caps
are retrospective controls.

## Comparisons

| Route | Method |
| --- | --- |
| Reuse | Original 20 owners and reuse-first scheduling |
| Legacy fitted | Original 20 and the published pair-score scheduler fitted on 20-person menus |
| Wide fitted, fixed | Original 20 and a pair model fitted on 60-person menus |
| Wide fitted, expanded | Evidence-trained selection of 20 from 60, followed by the same wide pair scheduler |
| Planned, fixed | Joint read plan restricted to the original 20 |
| Planned, joint | Joint read plan over 60, nominating the owners in the plan first and filling to 20 in original order |
| Joint nominees, open sources | Use the same planned nominees and let the pair scheduler select any of their sources |
| Adaptive, fixed or joint | Select sources and admit checked owners online, from either the original 20 or the full 60 |

The proposed plan greedily adds the document with the largest sum of
marginal estimated support over the available check quota. Allocate
the quota for each wave by dividing remaining checks by remaining
reads, rounding up. Cap each source's planned checks to that quota.
Discount an owner by the product of one minus their prior planned
support probabilities. This is an independent-outcomes heuristic.
At execution, use the same wide pair scheduler restricted to the
sources selected by the plan. Positive feedback removes subsequent
checks for an already supported owner. A pair score predicts observed
grade-2 support with grade-2 owner relevance. Missing judgments can
contribute zero observed credit, and are not verified negatives.

The two adaptive routes choose again after each observed check.
Compare the best available pair on open documents with the best
unopened document's estimated gain divided by its remaining-wave
check quota. Unused quota contributes zero to the estimate. A
new source wins only when this estimate exceeds the best open
pair or no open pair is available. Admit an expert when first
checked. Stop checking confirmed experts, and preserve the
read and check caps. The open-source and adaptive routes were
added after observing the initial six-arm comparison on the same
held-out topics.

## Results

| Route | Reads | Checks | Grade-2 supported / issued slots | Read oracle within nominated menus | Oracle inside planned sources |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reuse | 103 | 285 | 9 / 9 | 49 | — |
| Legacy fitted | 80 | 266 | 10 / 11 | 49 | — |
| Wide fitted, fixed | 78 | 282 | 10 / 12 | 49 | — |
| Wide fitted, expanded | 75 | 277 | 5 / 6 | 52 | — |
| Planned, fixed | 85 | 282 | 10 / 12 | 49 | 12 |
| Planned, joint | 81 | 285 | 8 / 10 | 50 | 10 |
| Joint nominees, open sources | 77 | 285 | 8 / 10 | 50 | — |
| Adaptive, fixed | 103 | 282 | 10 / 12 | 49 | — |
| Adaptive, joint | 102 | 285 | 7 / 9 | 50 | — |

The strongest earlier learned scheduler reproduces ten supported
slots. Training its pair model on wider menus or building a static
read plan for the original 20 reaches the same support count, with
more checks. At the same dispatch method, the wider fitted model
recovers five supported slots when it also changes the nominees.
A joint static plan recovers eight. Allowing that slate to reopen
its full source menus still recovers eight. The feedback-aware
joint route recovers seven against ten for the same feedback rule
with a fixed candidate universe. Relative to the legacy fitted route, paired P@5 differences are
−0.82 percentage points for the joint static plan (95% interval
−5.71 to +2.45) and −1.22 for adaptive joint (−6.12 to +2.45).
These intervals leave joint-method quality uncertain.

The read oracle counts the best supported output under each read
cap with access to held judgments. It omits the check cap. For a
planned arm the last column additionally restricts the oracle to
the selected documents. Neither oracle is a feasible policy.
Archive replays represent unknown pairs as unconfirmed rather than
assigning machine-verifier judgments. Local policy CPU time is
reported separately from outcome counts. It excludes topic encoding,
model fitting, document access and live verification.

Slurm tasks 46811_0 and 46811_1 on ws3 reproduce all results and
the held-topic comparison signature. The reuse and legacy fitted
routes reproduce the earlier published read, check and strict
support counts.

## Reproduce

Use the official topics and qrels with the pinned digests from
the [earlier scheduling comparison](../trec-coverage-scheduler-v1).
Run on a Slurm compute allocation and repeat with `--reverse`.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.trec_enterprise.joint_owner_source_plan TREC_DATA NEW_RESULTS
```

[Complete aggregate results](results.json)
