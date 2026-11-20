# Audit the opportunity for one more cross-source contribution check

Use the 130 verified PLOS XML records and 59 public request templates
from the [ORCID cross-source benchmark](../plos-credit-cross-source-v1).
Hold the first publication and both CRediT roles fixed. Read its
contribution declaration to establish the eligible author's identity.
Search for the second publication from its title sketch, automatically
edited abstract sentence, or abridged abstract sentence. A stress view
keeps alternating words from a central seven-word title span, removing
the contiguous match. Very short titles retain the original sketch. Run each request
in both source directions. These views share documents and are not
independent. The requested paper is part of the task. A different paper
with a valid role and the same person is a wrong-source link.

Rank at most eight candidates by notice-aware BM25 over titles and
abstracts. All routes inspect the first contender and one additional
contender from the same ranking. The two factors are whether to
prioritize source candidates whose indexed byline shares a verified
first-source contributor, and whether to use BM25 or ordered clue
alignment to rank the remaining candidates. If no candidate shares a
known identity, backfill by the corresponding topical ranking. Byline
metadata are available to the policy. A source's role declarations are
available only when that policy inspects it.

For the controlled blocked-source view, remove the requested second
paper before retrieval without rebuilding BM25 or changing the clue.
The visible no-link view keeps both papers, whose published eligible
role holders have disjoint identifiers. A structural support check
reads the selected publisher declaration and intersects the two roles'
eligible ORCIDs. Record extra true links and separate wrong-source
links, including when the target is withheld. Report how many initial
misses can be recovered by inspecting the correct source somewhere
among the remaining seven candidates.

A shared confirmation guard ranks the available shortlist by ordered
alignment. It gates a link on a unique leading source with sufficient
alignment and score margin. Fit a guard per clue style on the 104
non-target articles including their own withholdings. Select among the
existing source-confirmation thresholds, requiring zero observed wrong
confirmations on that calibration set. The target documents can appear
as distractors. If there is no feasible nonempty policy, the guard
abstains for that clue style. Report how many recoverable misses also
pass this guard. These empirical operating points do not establish
population-level error guarantees.

The source-first control tries the existing literal rule on notice-aware
metadata from the available catalog. A unique contiguous normalized
clue identifies a source. Otherwise use the same ordered guard. The
control reads at most one contribution declaration from the confirmed
source, and leaves unconfirmed sources unresolved. The source-first
arm uses source cues only at nomination and does not use ORCID for
document selection.

Read counts refer to original contribution declarations. Catalog
title/abstract scoring and byline access use indexed metadata.
This experiment audits retrieval and provenance, with supplied roles and
first witness. It does not measure open-ended expert matching or service
latency. Report paired percentile intervals by publication pair as
descriptive diagnostics. Some pairs share individual publications.

## Results

Two ws5 CPU tasks in Slurm array `44402` reproduced every
aggregate using distinct Python hash seeds. The 64 positive views
per style come from 32 queries over 11 publication pairs. All
routes except the source-first control read exactly two potential
second-source declarations per request.

| Clue style | Initial positive links / 64 | Extra possible in top eight | Extra under next BM25 | Extra under next ordered | Extra under owner then BM25 | Extra under owner then ordered | Correct source-first links / 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contiguous title sketch | 44 | 20 | 12 | 20 | 20 | 20 | 64 |
| Gapped title sketch | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| Edited abstract | 64 | 0 | 0 | 0 | 0 | 0 | 60 |
| Abridged abstract | 64 | 0 | 0 | 0 | 0 | 0 | 64 |

All 20 short-title misses are reachable in the common pool.
Owner binding recovers eight beyond the next-BM25 control and
adds none beyond the topical ordered control. Literal matching
avoids the second declaration read on these contiguous examples.
There is no feasible ordered guard for either short style on the
calibration inventory. Although BM25 retrieves the correct source
first for every gapped positive example, the available source-first
guard abstains throughout. This separates source ranking from
verifiable source confirmation.

With the gapped target withheld, an owner-prioritized second read
finds a structurally plausible link on four of 64 views, all on the
wrong source. Plain next-BM25 finds one and topical ordered finds
four. The source-first route leaves all 64 unresolved. The visible
no-link views have no incorrect raw links in any style. The results
favor assessing source-confirmation errors separately from the
contributor check. They do not establish an additional recovery
benefit over stronger document ranking.

[Verified aggregate results](results.json)

## Reproduce

Restore the 130 publisher XMLs listed in the prior benchmark's manifest
from the provided source URLs, retaining the exact SHA-256 versions.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.cross_source_gap_headroom \
  benchmarks/plos-credit-cross-source-v1 NEW_RESULTS XML_DIRECTORY
```
