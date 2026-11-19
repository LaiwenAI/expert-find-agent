# Select evidence for each person–work claim before a shared verifier batch

The [previous recheck experiment](../redocred-scope-recheck-v1)
found no qualifying full-context fallback at the development risk
ceiling. This experiment changes which sentences the verifier
reads for each claim. Use the same two people, two works, and one
role per document. Verify all four links independently in one
batch using the same frozen Qwen3-4B and prompt. Each route
submits exactly four pairs.

| Route | Input for one claim |
| --- | --- |
| Shared four | Up to four shared sentences chosen across both people, both works and the role |
| Lexical two | Two sentences with the best combined coverage of the requested person, work and role cues |
| Bound two | Two sentences prioritized by co-occurrence of the requested person and work and their proximity, with role and coverage cues |

All text is unedited. The two claim-specific routes select
from the full document. Exact name occurrences and partial
lexical matches use the same rules for both. The bound route
first orders pairs by same-sentence co-occurrence weighted by
role terms, plus a smaller adjacency term. It uses the lexical
coverage score next. The methods do not require exclusive
roles and never use human evidence spans during selection.

Fit one absolute threshold per route on the original development
cases, using 5% and 10% ceilings on accepted unjudged links.
After fitting, evaluate the previous 240-case train-document
panel and a fresh panel from the same upstream train release.
The fresh panel excludes every title in the previous panel and
all titles in the official development and test releases. Cohort
selection uses label structure and fixed hashes rather than
model output. Known claims come from the updated Re²-DocRED
release. Unjudged links remain unknown.

Report accepted known and unjudged links, full retention for
shared responsibilities, evidence word counts, and measured
batch P95. Measurements include sentence selection, tokenization
and the GPU forward. They omit source and candidate discovery,
model load, serving and queueing.

## Results

Slurm job 43359 completed on one RTX 5090 on ws5. It reproduces
the previous 5% compact baseline counts on all 240 cases. The
fresh panel contains 38 cases from 34 new titles. Its 116 known
and 36 unjudged links include 29 cases with shared roles. These
fresh cases are the remainder after the previous sample, so
the two test cohorts have different label mixtures.

| Panel and route | Known / unjudged accepted at 5% | Known / unjudged accepted at 10% | All shared links retained at 10% | P95 |
| --- | ---: | ---: | ---: | ---: |
| Previous, shared four | 245 / 24 | 304 / 32 | 31 / 120 | 70.8 ms |
| Previous, lexical two | 314 / 27 | 365 / 34 | 40 / 120 | 54.3 ms |
| Previous, bound two | 297 / 26 | 320 / 30 | 22 / 120 | 55.9 ms |
| Fresh, shared four | 59 / 5 | 64 / 5 | 11 / 29 | 68.7 ms |
| Fresh, lexical two | 65 / 5 | 83 / 6 | 13 / 29 | 53.9 ms |
| Fresh, bound two | 65 / 6 | 76 / 6 | 9 / 29 | 52.2 ms |

Ordinary claim-specific lexical selection approximately halves
the evidence word count and improves P95 by 21.5% on the fresh
panel. At the 5% calibration ceiling, it recovers six more
known links with the same count of unjudged acceptances. The
descriptive paired interval on confirmed links per four slots
is −0.6 to +8.3 percentage points. At the 10% ceiling, it
adds 19 known links and one unjudged link, with a descriptive
interval of +5.9 to +19.6 points.

Giving additional priority to same-sentence or adjacent names
does not improve complete shared-role recovery at either
calibration ceiling on either test panel. On the fresh 10%
slice it retains all shared links in 9 cases, versus 13 for
lexical two. Use the simple claim-specific selection as the
primary comparison for future within-source verification.
Keep the original shared evidence window available for
cross-sentence claims. Input identities, works and candidate
roles are supplied, and this diagnostic does not measure
end-to-end expert retrieval.
[Aggregate results and thresholds](results.json)

## Reproduce

Use the pinned original and extended releases specified in the
[scope recheck benchmark](../redocred-scope-recheck-v1) and the
same frozen Qwen3-4B-Instruct-2507. Run within a Slurm CUDA
allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.claim_bound_windows \
  REDOCRED_DIR RE2DOCRED_DIR QWEN4B NEW_RESULTS
```
