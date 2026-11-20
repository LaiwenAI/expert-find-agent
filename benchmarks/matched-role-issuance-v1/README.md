# Match issued recommendations before comparing role-first nomination

Use the frozen [PLOS](../plos-nomination-review-v1) and
[eLife](../elife-role-review-v1) citation packets and published digests.
No model parameters, retrieval settings, labels or rankings change.
For each aligned request, retain from both methods the prefix of
length equal to the smaller emitted count. Separate the difference
in tagged recommendations into the matched-prefix difference and
the difference in remaining slots. Within the latter, identify
requests where just one method issued a recommendation.

The unit of relevance is the reviewed topic–article pair. The same
label applies to all recommendation slots citing that pair, including
across both requests on the same topic. Count slots, not distinct
people or documents. For the matched comparison, the paired bootstrap
resamples whole topics and queries still have five possible slots in
the denominator. Also bound support differences exactly by varying
unjudged topic–article pairs, with and without assuming the topic tags
mark supported excerpts. These bounds contain no sampling uncertainty
and should not be added across prefix and tail because the same
case can occur in both.

## Results at full available declaration coverage

The full set of results also includes the sparse masks. Each row
reports the candidate minus comparator tagged counts. The final column
counts the part of the tail contributed by requests served by just one
method, with sign preserved.

| Panel and contrast | Issued slots | Total tagged difference | Matched prefix | Unmatched tail | Tail where the other method abstained |
| --- | ---: | ---: | ---: | ---: | ---: |
| PLOS topics, person four vs topic four | 68 vs 41 | +15 | +1 | +14 | +10 |
| PLOS topics, person four vs article-wide role union four | 68 vs 52 | +10 | +1 | +9 | +5 |
| eLife keywords, person four vs topic four | 126 vs 94 | +16 | +3 | +13 | +10 |
| eLife keywords, person four vs article-wide role union four | 126 vs 111 | +12 | +4 | +8 | +6 |
| eLife headings, person four vs topic four | 22 vs 22 | 0 | 0 | 0 | 0 |
| PLOS topics, top-eight person four vs topic eight | 54 vs 63 | −5 | +3 | −8 | −2 |
| eLife keywords, top-eight person four vs topic eight | 126 vs 141 | −4 | −1 | −3 | 0 |

In both four-versus-four topic contrasts, the matched-prefix interval
includes zero. At full available declaration coverage the descriptive
95% tagged P@5 differences for matched prefixes range from 0 to +1.30
percentage points in PLOS and 0 to +3.33 points in eLife keywords.
With all cases unjudged, the matched supported difference can range
from −3 to +3 slots for PLOS and −4 to +4 for eLife keywords.
Against topic four, the candidate serves six more PLOS requests and
nine more eLife keyword requests, accounting for ten additional
tagged slots in each panel. The remaining unmatched gains come from
deeper lists on requests already served by both methods.

The observed advantage of early person binding at fixed read budget
comes mostly from additional issuance. The prefix diagnostic provides
limited evidence of reordering benefit at the same result depth.
Match sizes depend on both realized outputs, so this analysis evaluates
the frozen lists and does not define a new deployable threshold policy.
No human labels were used. Report coverage and the support rate among
issued recommendations alongside fixed-denominator P@5 in future
independent evaluations.

[Aggregate decomposition](results.json)

```sh
PYTHONPATH=src:. python -m experiments.credit.matched_role_issuance \
  PLOS_ASSIGNER_PACKET benchmarks/plos-nomination-review-v1/results.json \
  ELIFE_ASSIGNER_PACKET benchmarks/elife-role-review-v1/results.json NEW_RESULTS.json
```
