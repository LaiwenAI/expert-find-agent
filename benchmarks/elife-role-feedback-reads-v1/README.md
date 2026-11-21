# Adapt the next paper read to a partial author-role proof

Transfer the scheduling question from the [TREC
source allocation replay](../trec-coverage-scheduler-v1) to two public
eLife panels. They require two different papers and the same author
with a separate publisher-declared role in each. This transfers the
allocation idea. No TREC model or fit parameters are reused.

Use the frozen [51-pair confirmation
panel](../elife-source-guard-confirmation-v1) and the earlier
[72-pair extended panel](../elife-extended-role-holdout-v1). Preserve
their document and author separation, matched role swaps, abstract
splits and known same-author proof labels. The initial index contains
titles, the first part of each abstract, and byline membership. Queries
use keywords or excerpts from the held-out second portion and the
original structured role labels. Use the existing frozen Qwen
embedding and reciprocal dense–lexical ranking, keeping twelve
documents per slot. All policies see the same candidates and linked
authors. Only the act of reading a document reveals its role declarations.

| Policy | Allocation |
| --- | --- |
| Round robin | Interleave the two topic rankings and read the next distinct document. |
| Static byline | Read the distinct same-author pair with best worst rank, breaking ties by summed rank. Spend any additional reads in round-robin order. |
| Role feedback | Start with the same first read as static byline. After each read, rule out pairs that cannot complete any author's two roles and prefer feasible pairs needing the fewest further reads. |
| Expected proof | Prioritize a pair's estimated proof yield per unread document. Multiply reciprocal topic ranks on unread sides and combine the signal over possible same-author links. Condition on exact roles already read. The score is a rank heuristic with no calibrated interpretation. |

Run with exactly two, three or four distinct reads. For every policy
return the best-ranked fully supported pair among opened documents.
Either keep all byline-compatible pairs or apply the same frozen top-3
directional source guard before allocation and issuance. Under source
withholding, the intended paper is absent from retrieval and reads.
A strict success requires the exact target pair and one gold author.
A returned substitute or a returned proof on a role-swap query is
counted as adverse for this exact-source task.

The upper limit counts positive cases whose exact target pair is
available in the shared menu before allocation. The interval resamples
paper/author connected groups. Each panel runs twice in reverse family
order. This is a search-policy replay with a declaration oracle. It
charges logical declaration reads without modeling document I/O,
role parsing errors, or complete service latency.

## Same-author allocation results

At two reads, the static byline plan saturates the exact-pair
opportunity in both panels and for both input views. Neither
feedback policy recovers an additional strict proof at any
tested read cap. The table shows the stricter keyword condition.

| Keyword panel with source guard | Positive requests | Round robin, two reads | Static byline, two reads | Role feedback, two reads | Exact pairs in menu |
| --- | ---: | ---: | ---: | ---: | ---: |
| Confirmation | 51 | 29 | 39 | 39 | 39 |
| Extended | 72 | 43 | 52 | 52 | 52 |

In the guarded keyword condition, both panels have zero adverse
outputs with both papers present or one withheld. With the guard
removed and three reads, static and feedback still tie on strict
proofs, but the feedback policy increases adverse keyword outputs
from 3 to 4 in confirmation and from 6 to 8 in extended. With the
guard and an excerpt clue, a withheld target can still produce a
substitute. At two reads the static method does so in 3 of 204
confirmation controls and 6 of 288 extended controls.

This is a ceiling result for the available menus under exact
declaration decoding. Further adaptive reads have no positive
headroom in this replay. The separate fixed-source experiment below
relaxes the perfect role decoding assumption.

## Role recognition with the source pair fixed

Use the same static pair chosen under the top-3 directional guard.
As a control, supplying the intended structured role must reproduce
the two-read static outcome. Replace the known role with its original
natural-language phrase from the frozen [NISO
examples](https://zenodo.org/records/18421449). Compare the frozen
ridge and prototype classifiers with the existing open 4B paired
reader. That reader gets either the three candidates proposed by the
combined embedding scores or all 14 roles with a no-match option.
Reuse the earlier cutoff. No eLife labels tune the models.

Each decode is independent of document availability and reused
across keyword, excerpt and withholding versions of the same role
request. Report logical forwards and input tokens assuming
independent requests. The experiment does not measure latency or
claim caching speedups. A strict success also requires the roles
cited by the system to match the requested roles.

| Complete catalog, keyword clues | Confirmation strict / 51 | Confirmation adverse / 102 | Extended strict / 72 | Extended adverse / 144 |
| --- | ---: | ---: | ---: | ---: |
| Exact role reference | 39 | 0 | 52 | 0 |
| Ridge top one | 24 | 10 | 30 | 5 |
| Prototype top one | 25 | 9 | 35 | 3 |
| 4B top three | 27 | 5 | 38 | 4 |
| 4B all roles | 29 | 4 | 39 | 8 |

At the identical source pairs, full-role 4B decoding improves
over the ridge by five and nine strict keyword proofs. The
connected-group interval spans zero in the confirmation panel
(−2.1 to +22.2 percentage points) and stays above zero in the
extended panel (+3.2 to +21.9). Compared with the three-role menu,
all-role decoding spends roughly 72% more input tokens for two
and one additional strict proofs. Adverse outputs fall from five
to four in confirmation and increase from four to eight in extended.
Both retain zero withheld-source substitute issuance for keyword
queries under the existing source guard.

Exact role decoding on the same source pairs reproduces every
aggregate of the guarded static two-read reference. With natural
phrases, inference still leaves a gap of 10 and 13 strict keyword
proofs. At this fixed read budget the actionable error comes from
role recognition and authorization of the resulting proof.
In the excerpt condition, a correct role decoder can still issue
a wrong-pair substitution if one source is absent.

All eight GPU subjobs completed. Within each panel and comparison,
both family orders reproduce the reported outcomes and output
signatures. [Full aggregates](results.json)

## Reproduce

Restore both pinned projections and manifests from the linked
publisher experiments. Use the same model checkpoint in a Slurm GPU
allocation and repeat with `--reverse` and a fresh results path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.elife_role_feedback_reads PROJECTION MANIFEST QWEN_EMBEDDING NEW_RESULT
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.elife_fixed_pair_role_gap \
  PROJECTION MANIFEST QWEN_EMBEDDING ROLE_CHECKPOINT NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/paired-role-verification-v1/results.json NEW_ROLE_RESULT
```
