# Calibrate the reader on the candidates it will see during rescue

In the [fixed FollowIR rescue replay](../followir-constraint-rescue-scope-v1),
the 0.6B reader rejects every judged positive nominated by broader rescue.
Test whether the topical calibration distribution accounts for the rejects.
Keep the model, 850-character lead excerpts, first four documents, initial
acceptance threshold and two full-pool rescue candidates unchanged.

In each direction, transfer from the other FollowIR collection. For every
topic in that calibration collection, compute the initial 10% negative cap
from all other calibration topics, and use it to decide whether to open
rescue on the excluded topic. This produces out-of-fold rescue candidates
for the calibration collection. The first-pass threshold used at test time
uses all calibration topics, as in the original protocol.

Compare three ways of scoring the same two test candidates:

- **Topical cutoff:** calibrate the raw margin on the other collection's
  first six conventional topic candidates.
- **Rescue cutoff:** calibrate the raw margin on cross-fitted rescue
  candidates from that collection.
- **Relative rescue cutoff:** calibrate the margin above the maximum
  first-batch margin on the same cross-fitted rescue candidates.

For each route use 5%, 10% and 20% observed known-negative acceptance
caps on its calibration sample. The primary operating point is 10%.
The three test policies add no reader calls. Report document positives,
known negatives, previously relevant documents invalidated by the
changed constraint, and unjudged acceptances. Fit no parameters on
the target collection. A calibration cap is an empirical target and
does not imply a matching test risk. Paired intervals resample topics
with a rescue action, so they do not describe all first queries.

## GPU results

Job `43599` reproduced the first-pass thresholds and rescue-candidate
counts on one RTX 5090. Cross-fitting yields 14 known-negative rescue
examples in Core17 and 25 in News21. The 10% calibration targets allow
one and two negative acceptances, respectively. On each transferred
test set, the original topical threshold accepts no rescue candidates.

| Transfer | Test positives / negatives | Topical accepted + / − | Rescue accepted + / − | Relative rescue accepted + / − |
| --- | ---: | ---: | ---: | ---: |
| Core17 to News21 | 11 / 17 | 0 / 0 | 3 / 5 | 0 / 0 |
| News21 to Core17 | 13 / 17 | 0 / 0 | 4 / 0 | 4 / 3 |

Raw rescue calibration changes the cutoff from 3.25 to 0.375 in
the first direction and from 4.75 to 2.50 in the reverse direction.
At the former cutoff, the test accepts 5/17 known negatives, including
two documents invalidated by the changed instruction. The reverse
direction's 0/17 known-negative acceptances are an observed count,
with substantial uncertainty given the denominator. No unjudged
document is accepted at these operating points.

Rescue calibration can recover some positives without changing
the model or calls. Transfer does not preserve the calibration
risk target. Subtracting the initial maximum fails to improve
both transfers. The borrowed cutoff explains part of the original
rejection pattern, but the small, reused document cohort does not
justify a common production threshold. Collect target-domain
negative rescue examples and separate constraint violations from
missing evidence before evaluating a default gate.

[Aggregate results and sensitivity points](results.json)

```sh
PYTHONPATH=src:. python -m experiments.followir.rescue_threshold_transfer \
  CORE17_JSON CORE17_VECTORS NEWS21_JSON NEWS21_VECTORS EMBEDDING RERANKER \
  PRIOR_RESCUE_RESULTS NEW_RESULTS
```
