# Shortlist and repair a proposed duty before binding authors

Compare five-head semantic veto, two-head veto, and two-head repair without
reopening source search. Reuse the [semantic confirmation benchmark](../plos-open-role-semantic-confirmation-v1)
and its 197-article PLOS catalog. All arms share the same query encoder, source
nomination and confirmation, and out-of-inventory gate.

The shortlist is the first two roles ordered by the existing embedding scores
against enriched role definitions. It needs no extra embedding. One frozen
Qwen3-Reranker-0.6B batch scores both canonical role headings. A role's score is
the minimum of its yes/no logit difference and its lead over the competing
heading. The largest incorrect shortlist score on the five earlier authored
paraphrases and five outside activities sets the cutoff. The five correct roles
appear in the calibration shortlists and all pass a positive margin.

| Policy | What triggers verification | Allowed action |
| --- | --- | --- |
| Five-head veto | The frozen baseline would recommend an author | Keep the original proposal or abstain |
| Two-head veto | The frozen baseline would recommend an author | Keep the original proposal or abstain |
| Two-head repair | Both gates pass and at least one distinct shortlisted duty shares a credited author with the first duty | Bind the semantic winner when its score is strictly greater than both zero and the calibration cutoff, it differs from the named first duty, and an author is credited with both |

The verifier sees the request and the headings. Author eligibility is checked
from the publisher's fields; it cannot make a semantically losing heading win.
Repair can switch the proposed role even when no author was credited with the
old proposal. All thresholds were set without fitting to the evaluation phrases.

## Results

Pilot and balanced-order runs (Slurm jobs 42882 and 42884) produced identical
counts across all panels. Job 42884 completed on an RTX 5090. Both historical
controls replay their frozen outcome counts. Each table row covers 72 supported
queries when the intended article is in the catalog.

| Clue | Policy | Clean | Wrong duty | Wrong source | Component P95, ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Excerpt | Frozen proposal | 18 | 13 | 0 | 15.20 |
| Excerpt | Five-head veto | 13 | 0 | 0 | 23.79 |
| Excerpt | Two-head veto | 13 | 0 | 0 | 23.70 |
| Excerpt | Two-head repair | 17 | 0 | 0 | 23.70 |
| Keywords | Frozen proposal | 14 | 9 | 1 | 14.49 |
| Keywords | Five-head veto | 10 | 0 | 0 | 23.18 |
| Keywords | Two-head veto | 10 | 0 | 0 | 23.01 |
| Keywords | Two-head repair | 14 | 0 | 0 | 23.02 |

Repair changes the proposed duty on 13 clean excerpt and nine clean keyword
responses. These replacement counts differ from the net gain of four clean
responses per clue relative to veto: the positive-margin repair policy can also
abstain on some previously clean cases. The gated, structurally valid shortlist
contains the correct duty on 31 excerpt and 23 keyword requests, yielding 17/31
and 14/23 clean returns within that reachable set. When the article is withheld,
the two veto policies each return one wrong source in the keyword condition;
repair returns none. Every policy abstains on the tested outside activities.

Scoring two headings rather than five saves about 0.06/0.20 ms conditional
verification P95 for excerpt/keyword requests. Both still pay for a model
forward, and the observed component difference is small. The measurement
charges each policy for its own path, rotates verifier order, and omits
index construction, transport, and queueing. It does not measure end-to-end
service latency.

On this catalog, constrained repair retains more correct results than veto
alone without introducing a wrong recommendation. Repair still costs about
8.5 ms above the frozen reference and depends on exact contributor statements.
Evaluation wording repeats across sources. Corpus-wide figures cannot be
interpreted as transfer to unseen duty phrasing or unstructured enterprise
evidence.

[Aggregate results](results.json)

## Reproduce

Use the command and inputs from the
[semantic confirmation benchmark](../plos-open-role-semantic-confirmation-v1).
Use a fresh output path and append:

```sh
--shortlist-reference benchmarks/plos-open-role-semantic-confirmation-v1/results.json
```
