# Separate identity nomination from verification in first-query expert search

Use the [frozen 452-article mixed corpus](../owner-evidence-bound-v1)
and its 160 PLOS ONE and 51 eLife requests. Each request requires two
topic–role conditions to be satisfied by the same person on two distinct
articles. Compare identity hints at a fixed two-declaration read budget.

For each of three deterministic masks, hide author identifiers in 25%
or 50% of source records at index time. A masked source retains its
declared roles and author ordinals, but its identity fields do not
participate in the index. The original declarations remain available
when a selected source is read. Naturally missing identities remain
unresolved. The full-name condition uses each publication's normalized
name. The controlled abbreviated condition replaces the target's
matching key with first initial and family name, on both the candidate
and reference side. Exact full names are still used for other byline
members in the collaborator signal. These are simulated index visibility
and name variants, not estimates of their natural prevalence.

| Index view | Candidate identity |
| --- | --- |
| Visible identifiers | Publisher identifiers available under the mask |
| Unique name | Visible identity from another source in the same publisher with a matching name key |
| Name plus coauthor | Matching key and at least one shared coauthor |
| All identifiers | Unmasked publisher identifiers, as a reference |

Name routes assign a tentative identifier only if exactly one compatible
known identity remains. Conflicting assignments of one person to multiple
author rows in the same source are rejected. Tentative identities never
become anchors for further linking. A static author-role join chooses
the two sources under the same lexical and Qwen3-Embedding-0.6B topic
rankings. When that join returns no pair, use the ordinary topic
fallback. The read path evaluates both the unsafe diagnostic (exact
roles assigned to the tentatively linked author ordinal) and the
source-verified result (original publisher roles and identifiers read
together). Both outputs use only the same two articles. A verified
result is additionally counted as known when both articles carry the
publisher topic annotations. Missing topic annotations remain unresolved.

## Results at candidate depth 128 and 50% nominal masking

Ranges are across the three masks, with fixed requests and corpus. The
unmasked reference returns 85 known matches on PLOS and 46 on eLife.

| Name visibility and policy | PLOS known after reading | eLife known after reading | PLOS confirmed wrong if links are trusted | eLife confirmed wrong if links are trusted |
| --- | ---: | ---: | ---: | ---: |
| Visible identifiers only | 9–39 | 0–18 | 0 | 0 |
| Full name, unique | 73–75 | 25–34 | 0 | 0 |
| Initial and surname, unique | 54–58 | 21–28 | 2–46 | 0–20 |
| Name plus shared coauthor | 44–49 | 14–32 | 0 | 0 |

The coauthor policy has matching results under both name modes in this
fixture. In the most ambiguous abbreviated run, the offline name-only
index creates eight incorrect author links with known contrary labels
and 21 unresolved links. Across reuse by 211 requests, 46 PLOS and
20 eLife recommendations cite an author with a conflicting identity.
Role labels are read accurately in that diagnostic. The error comes
from assigning the role to the guessed person. Checking the original
identifiers on those same sources removes the wrong outputs. Even
with full names, accepting tentative links directly yields 5–10
identity-unresolved PLOS outputs per mask. Those unresolved cases
do not count as verified matches or confirmed wrong matches.

Treat tentative name matches as source nominations and re-check the
author identity before using a source as evidence of a person's
qualifications. The publisher identifier provides the independent check
in this replay because its original value is retained in the selected
source. The benefit cannot be assumed where original identity evidence
is absent, and the observed zero known wrong links with collaborators
is not a statistical guarantee. The materialized claims here are
publisher-declared roles on structured inputs. This test does not
measure enterprise eligibility, network latency, or the cost of
acquiring new identity evidence. The P95 diagnostics include the
joint replay of both read output modes.

## Reproduce

Use the publisher XML manifests and the frozen embedding checkpoint
from the [declaration-read study](../owner-evidence-bound-v1).
Slurm GPU jobs 45599 and 45606 on ws3 reproduced the nontiming results
at both candidate depths. The result files include the full four
visibility settings per publisher and paired topic-group diagnostics.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false python -m experiments.credit.identity_role_search \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json \
  NEW_RESULTS --depth 128
```
