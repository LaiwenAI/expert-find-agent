# Hold out newer eLife papers for role prompt transfer

The earlier [compact prompt study](../compact-role-prompts-v1)
reused the engineering and eLife development panels. This test
uses new articles from the same pinned [publisher XML repository](https://github.com/elifesciences/elife-article-xml/tree/58250dd64e27afae6bc2700bcddd84dd38776e0d).
Select article IDs in `[65000, 84000)`, in SHA-256 order of
`expert-compact-transfer-20261121:<id>`. Take the first 512,
using each ID's highest version at that revision.

Require eligible licenses and article metadata under the prior
eLife parser. Exclude exact DOI, normalized title, normalized
author name and checksum-valid ORCID overlap with the local
earlier source inventories. The audit covers the prior eLife,
engineering and PLOS XML snapshots available for screening.
Only sources with complete role declarations, unique author
keys and a split abstract enter the search index. Author links
within the new pool require the same valid ORCID. Build paired
scope swaps using the existing engineering construction.

Keep the same static prototype and guarded 4B recheck arms.
The rendering of the candidate role options is the only change
across the three reader versions. Use the frozen models,
definitions, role bands, margin cutoff and NISO wording bank.
Evaluate excerpt and keyword clues, with both target papers
visible and with each target separately removed. The wording
bank is reused from development. The holdout tests source
and author transfer under known clause boundaries.

## Results

All 512 source files match their Git blob IDs. The prior audit
covers 1,009 distinct XML snapshots and excludes 64 held records
with author overlap. Twenty-seven other records fail the
publisher metadata and research article filter. The 421 retained
records reproduce the pinned projection byte for byte. After
the declaration filter, 393 articles enter the index. The
test has 66 paired positives and 66 scope swaps spanning 13
article pairs and 14 credited contributors.

| Clue | Policy | Strict proofs / 66 | Adverse / 132 | Prompt tokens | Local P95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Excerpt | Static prototype | 35 | 4 | 0 | 15.5–15.6 |
| Excerpt | Full | 41 | 6 | 7,733 | 38.4–38.7 |
| Excerpt | Compact | 40 | 6 | 4,848 | 29.1–29.5 |
| Excerpt | Name only | 42 | 6 | 3,074 | 26.5 |
| Keywords | Static prototype | 26 | 1 | 0 | 10.8–11.2 |
| Keywords | Full | 32 | 1 | 4,567 | 33.0–33.5 |
| Keywords | Compact | 32 | 1 | 2,882 | 24.9–25.2 |
| Keywords | Name only | 33 | 1 | 1,834 | 21.5–22.0 |

Adverse results include either a wrong contributor, cited role,
or selected source. Each reader route makes 32 excerpt calls
and 19 keyword calls. All four routes show zero selected-source
substitutions in each set of 264 missing-source views. The
name-only variant cuts about 60% of prompt tokens compared with
full definitions and finds one additional strict proof in each
clue view. All prompt variants share the same adverse counts
in this holdout. In contrast, the earlier eLife cohort observed
additional errors with short prompts. Report the two cohorts
separately. The finding establishes a latency difference for
this test and does not establish equivalent population risk.

Both tasks in Slurm array 45957 ran on ws3 RTX 5090 with reversed
request order and reproduced all non-timing fields. The local
factored runtime includes warm embedding, source retrieval,
role checks and the selected reader call. Construction, service
latency and free-form parsing are outside the timer. The cases
repeat roles and articles within 13 paper pairs. Reported error
counts are descriptive. [Results](results.json)

## Reproduce

Download the retained XML files in [the manifest](manifest.json)
from the pinned publisher revision's `articles` directory.
The original preparation utility checks source hashes and
restores the exact projection. Run the comparison inside a
Slurm CUDA allocation. Use a fresh result path each time.

```sh
PYTHONPATH=src:. python -m experiments.credit.elife_role_nomination prepare \
  ELIFE_XML benchmarks/elife-prompt-holdout-v1/manifest.json ELIFE_PROJECTION
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.elife_prompt_holdout run \
  ELIFE_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```

Repeat with `--reverse`. The manifest holds every sample
filename, its decision, its SHA-256 and the reference audit
digest. The source files and abstracts are not duplicated in
this release.
