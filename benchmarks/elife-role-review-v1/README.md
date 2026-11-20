# Separate recommendation issuance from topical support on eLife

Reproduce the [eLife nomination experiment](../elife-role-nomination-v1)
with exact ordered cited DOIs. CUDA job `43772` matches all frozen
non-timing results from both earlier runs. Replay validation also
recounts issued, tagged and off-label recommendation slots directly
from the trace for each panel, declaration mask and method.

For each method, count how often it cites each article for a given
topic. A review case asks whether the exact title and abstract support
that topic. Give each topic–article pair one judgment even when several
expert slots cite it. The existing structured declaration continues to
establish who reports each role. Topical support in the excerpt cannot
establish broader expertise, author contribution quality, or enterprise
suitability. The blank review forms hide route, rank, keyword-tag
status, DOI and priority. Identical decisive judgments from two
reviewers count as settled. Missing, unclear and disputed judgments
remain open. No human decisions were available for this release.

The total of 82 distinct cases covers both panels and both masks.
Within the full-coverage keyword panel, 30 distinct pairs have
nonzero net citation counts in at least one of the three contrasts
below. These are the priority for a paired difference in supported
slots. The support fraction among issued recommendations uses
different denominators, so even citations emitted by both methods
can affect its contrast. All 82 cases are available for independent
review.

## Exact bounds on the 48 keyword requests

For each case, subtract the comparator's citation count from the
candidate's count. Sum those weights over supported cases. For the
support fraction conditional on issuance, first divide each method's
count by its own fixed total issuance and then subtract. For any
unjudged case, assign its support to minimize or maximize that sum.
These are exact finite-cohort identification bounds. They include no
sampling uncertainty. The second scenario assumes tagged cases are
supported and lets untagged cases vary.

| Full available declaration coverage | Issued slots | Supported slot difference, all unjudged | Supported slot difference, tagged cases supported | Conditional support difference, tagged cases supported |
| --- | ---: | ---: | ---: | ---: |
| Person four vs topic four | 126 vs 94 | 0 to +32 | +16 to +32 | −12.70 to +8.11 pp |
| Person four vs article-wide role union four | 126 vs 111 | 0 to +15 | +12 to +15 | −2.38 to +4.61 pp |
| Top-eight person four vs topic eight | 126 vs 141 | −19 to +4 | −17 to −2 | −5.30 to +9.22 pp |

Slot differences use the fixed denominator of 240 positions.
Against topic four, the person route emits 32 additional slots. Its
topic–article counts never decrease after aggregating the two role
queries for each keyword. Even if all tags are accepted, a higher
number of supported recommendations can coexist with a lower
support fraction among the recommendations issued. Against topic
eight, accepting all tagged cases leaves at least a two-slot deficit
but allows a much larger one. None of these bounds establishes a
positive conditional support benefit or changes the frozen
noninferiority conclusion. The complete results include both label
scenarios for both panels and masks.

[Aggregate audit](results.json)

## Reproduce

Rebuild the pinned public corpus and use the same Slurm instructions
as the original experiment. In the original run command add
`--witness-output NEW_WITNESS_FILE`. Then prepare the private
assigner packet and separate blank forms locally.

```sh
export PYTHONPATH=src:.
python -m experiments.credit.elife_role_review prepare PROJECTION REPLAY_RESULTS \
  WITNESS_FILE benchmarks/elife-role-nomination-v1/results.json NEW_REVIEW_DIR
python -m experiments.credit.elife_role_review reconcile NEW_REVIEW_DIR/packet.json \
  COMPLETED_REVIEWER_A COMPLETED_REVIEWER_B NEW_ADJUDICATION_FILE
```
