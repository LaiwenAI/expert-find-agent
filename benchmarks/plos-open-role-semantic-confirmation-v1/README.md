# Confirm proposed person-specific duties in open publisher-note search

A source that mentions a person can still show a different duty from the one
requested. This experiment measures whether an independent semantic check can
reject that mistake while retaining correct recommendations.

Reuse the [open source and role benchmark](../plos-open-role-applicability-v1).
It searches 197 PLOS articles, with 72 matched exact-study requests from 70
author groups. The two clues are an abstract excerpt and keywords. The intended
study can be present or withheld, and requested duties can be supported or
outside the legacy role inventory. Labels come from publication identity and
structured contributor declarations. The activity wording comes from selected
NISO guide examples and repeats across papers.

Freeze hybrid source nomination, the full-text source fallback, the enriched
role proposal, and the exact intersection of credited authors. The verifier runs
only when this reference policy would return a recommendation. It sees the
short requested activity and canonical legacy headings. It receives no name or
source text, never changes the proposed role, and may only veto.

- **Single head:** score the proposed heading with the open Qwen3-Reranker-0.6B.
- **Rival heads:** score all five headings in one forward pass. Use the minimum
  of the proposed heading's yes/no logit difference and its lead over the best
  competing heading.

Both cuts are frozen at the maximum incorrect-pair score on five previously
authored role paraphrases and five outside activities, crossed with all five
headings. The five correct pairs survive both cuts. This small calibration set
is disjoint in wording from the NISO evaluation. Text-to-label semantic matching
is an established comparison mechanism. [De Silva et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.475/)

## Results

Slurm job 42878 completed on an RTX 5090. Control outcome counts match the frozen
baseline in every panel. The table reports supported requests with the intended
study in the catalog, 72 per clue form. A wrong source is counted separately
from attributing the wrong role on the correct study.

| Clue | Policy | Clean | Wrong role | Wrong source | Component P95, ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Abstract excerpt | Frozen reference | 18 | 13 | 0 | 15.18 |
| Abstract excerpt | Single head | 4 | 0 | 0 | 22.20 |
| Abstract excerpt | Rival heads | 13 | 0 | 0 | 23.70 |
| Keywords | Frozen reference | 14 | 9 | 1 | 14.48 |
| Keywords | Single head | 5 | 0 | 0 | 21.61 |
| Keywords | Rival heads | 10 | 0 | 0 | 23.10 |

At the rival-head verifier's observed return counts, keeping the highest
initial proposal margins would retain 9 of 13 clean excerpt responses and 5
of 10 clean keyword responses. Rival heads retain 13 of 13 and 10 of 10. Ties
cannot change these clean counts. At the single-head verifier's observed
coverage, the initial margins alone retain the same number of clean responses
with no errors. The matched-return control uses observed return counts for
diagnosis and does not define a prospective threshold.

Each verifier runs on 31 excerpt and 24 keyword supported requests with the
target present. Rival-head conditional verification P95 is 8.93 and 8.98 ms.
Timing sums live component times along each request path, with alternate verifier
order and no verdict cache. It excludes model loading, index construction,
transport, and queueing; it does not measure service latency. When the intended
study is withheld, both verifiers see the same two keyword false accepts. The
rival check leaves one wrong-source response and the single-head check leaves
none. All arms abstain on the tested outside-inventory requests.

The independent heading check improves selection at matched coverage in this
controlled catalog. It still rejects five clean excerpt responses and four
clean keyword responses compared with the frozen operating policy. Published
notes provide exact author scope, and the study uses repeated role wordings.
These results therefore motivate testing cheaper contrastive role scores on
independent wording and corpora. They do not establish personal contribution
from free text or better end-to-end expert matching.

[Aggregate results](results.json)

## Reproduce

Use the pinned captures, NISO file, calibration artifacts and embedding specified
in the [open source and role benchmark](../plos-open-role-applicability-v1),
plus its result JSON and the pinned reranker. Run inside a Slurm CUDA allocation
with a fresh output path.

```sh
PYTHONPATH=src:. python -m experiments.credit.open_role_semantic_confirmation \
  OLD_PUBLISHER_SNAPSHOT FIRST_TRANSFER_SNAPSHOT FRESH_TRANSFER_SNAPSHOT NISO_EXAMPLES \
  ROLE_APPLICABILITY_RESULTS OPEN_SOURCE_RESULTS BODY_CONFIRMATION_RESULTS \
  OPEN_ROLE_RESULTS EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_RESULTS.json
```
