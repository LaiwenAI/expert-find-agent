# Probe the author index when one source needs recovery

The [nested source study](../elife-nested-source-acceptance-v1)
found that symmetric expansion from six to 18 often spent extra
work on withdrawn sources and returned no additional verified
results after source gating. This follow-up tests two smaller
search spaces tied to a plausible first source.

## Procedure

Use the same 2,011 eLife papers and 35 author-disjoint held groups.
The inputs are excerpt clues or keyword clues with two contribution
conditions. The frozen query-first role decoder must first be
confident in both roles. If the depth-six linked source pair supports
those roles for the same contributor, return it. On a confident role
request with no such proof, compare three fallbacks.

| Fallback | Discovery space |
| --- | --- |
| Global | All pairs from the two top-18 hybrid lists |
| Author | Take the first hybrid source as an anchor on each side. For the other condition, search the union of its authors' paper postings and keep 12 local hybrid candidates per anchor. |
| Author and role | Search the opposite role posting only for authors declared to hold the first role on the anchor. Everything else is shared with the author-only route. |

An anchor can be a wrong source. Both papers therefore go through
the same learned pair score and the same-author role requirement.
An author-and-role posting is a precomputed index over resolved
identities and structured declarations, not an inferred role label.
A locally ranked paper can lie outside the global top 18. Reject
a withdrawn DOI before ranking within the postings.

Share a logistic source-pair classifier trained on proposals from
18 old development groups. It uses absolute, relative and
cross-clue cosine scores plus weighted lexical footprint coverage.
Fit on both global linked candidates and each method's proposals;
deduplicate pairs within each paper-group/availability case.
Calibrate each fallback separately using the other 42 historical
groups. Set its threshold at the third-highest group maximum among
potential wrong exact proofs, using strict greater-than acceptance.
This calibration includes misinterpreted roles and substituted
sources, and changes none of the shared shallow outputs.
All evaluation groups remain separate from fitting and calibration,
and have been explored in the previous source experiment.

Count strict proof recovery, wrong issuance, source rows ranked,
author index entries visited and candidate pairs scored. Multiple
role requests share the same paper groups. Work counters are per
request and count a duplicate visited from each side separately.
The replay uses resident dense scores and precomputes global ranks
for comparison. It measures candidate work and quality, not
end-to-end time or GPU service capacity.

## Results

| Entrance and route | Correct / wrong, sources present | Wrong, source withdrawn | Source rows on present requests | Posting entries on present requests |
| --- | ---: | ---: | ---: | ---: |
| Excerpt, depth-six linked | 38 / 0 | 1 | 0 | 0 |
| Excerpt, global fallback | 40 / 7 | 5 | 1,512 | 0 |
| Excerpt, author fallback | 40 / 3 | 1 | 91 | 472 |
| Excerpt, author-role fallback | 40 / 3 | 1 | 8 | 115 |
| Keyword, depth-six linked | 35 / 0 | 0 | 0 | 0 |
| Keyword, global fallback | 37 / 8 | 8 | 1,620 | 0 |
| Keyword, author fallback | 37 / 3 | 0 | 69 | 415 |
| Keyword, author-role fallback | 37 / 3 | 0 | 9 | 112 |

Each entrance has 204 present-source requests and 408 requests
with a source withdrawn. Work columns count fallback only.
Relative to author-only access, adding the requested role reduces
the number of source rows ranked from 91 to 8 for excerpts and 69 to 9
for keywords. The number of definitive pair scores is unchanged at
eight and nine, and the output counts and calibrated cutoffs match.
Posting visits drop from 472 to 115 and 415 to 112 with both sources
available, and from 1,501 to 511 and 1,370 to 405 after withdrawal.
The stronger comparison is the pair of author-index arms, which
perform the same kind of indexed access. The global row count is
the number of entries touched in an already-ranked list.

Relative to the shallow control, both index routes add two correct
and three wrong proofs in each present-source view. They introduce
no new wrong output after source withdrawal. The two extra correct
proofs are also reachable in the global shortlist. The global
keyword cutoff is inactive because only two calibration groups
have any adverse fallback proposal, below the number the policy
allows. Its higher observed error count does not isolate an effect
of candidate scope at matched held risk. The fit also had only one
distinct wrong keyword pair. Treat the result as an index selectivity
finding and an error-budget warning. A replacement policy needs
verification in another corpus and better acceptance evidence for
present-source cases.

Both tasks of Slurm array 47157 on ws6 (RTX 4090) agree exactly,
and the depth-six responses reproduce archived counts.
[Aggregate results](results.json).

## Reproduce

Restore the public inputs and checkpoint as in the
[previous source study](../elife-nested-source-acceptance-v1).
On a Slurm GPU allocation, run with a fresh output path:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.directed_source_rescue \
  CONFIRMATION_PROJECTION EXTENDED_PROJECTION \
  benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json \
  benchmarks/elife-held-source-acceptance-v1/panel.json \
  benchmarks/role-first-proof-decoding-v1/results.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json QWEN_EMBEDDING NEW_RESULTS
```

Repeat with `--reverse` and a fresh results path.
