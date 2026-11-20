# Transfer author-scoped verification across engineering publishers

Use 2025 engineering articles from *Sensors* and *Frontiers in
Robotics and AI*. Take the first 100 Europe PMC search results per
ISSN with open-access status and hash-select 60 PMCIDs per journal.
Fetch the journal records with the official PMC EFetch interface.
Match DOIs and archive the exact XML projections. Verify individual
CC BY licenses before using an article.
[Europe PMC API](https://dev.europepmc.org/RestfulWebService) ·
[PMC developer documentation](https://pmc.ncbi.nlm.nih.gov/tools/developers/)

## Construct independent representations

The structured reference reads author-level JATS role tags. The
visible reference reads only the rendered author-contribution text
and byline. One parser reads role-first, semicolon-separated lists
in *Sensors*, and the other reads author-first blocks in *Frontiers*.
Normalize terms to the 14 CRediT categories. Resolve initials against
the byline before comparing role assignments. Reject ambiguous names,
unknown roles, incomplete authors and disagreements. The reference
requires exact agreement for every author in an included paper.
This avoids inferring an absent role from one incomplete record.
The two representations still come from the same publisher and do
not independently verify real contributions.
[CRediT role taxonomy](https://credit.niso.org/contributor-roles/)

From 120 source articles, retain 14 *Sensors* and 40 *Frontiers*
articles. The source audit records every exclusion. There are 14
disagreements among the 68 articles that reached the complete
role-set comparison. None is used as a model error or training label.
There are no exact normalized byline names shared across the included
journals. More subtle author identity overlap is possible.

## Verification

For each declared role with at least one credited author and one
uncredited coauthor, choose one of each by stable hash. Test the same
pair with the exact role name and one hand-authored task description
for that role. The descriptions repeat across papers. The verifier
receives the visible statement, with each candidate identified by
their original initials. The scope rule uses contributor mentions
resolved from the declaration and byline, without using role labels
to decide which blocks to keep.

- **Shared:** show the full declaration to both candidates.
- **Length control:** keep the same number of blocks as candidate
  scope, taken from the beginning of the declaration.
- **Candidate scope:** keep the blocks mentioning the target author.
- **Scoped role only:** keep the same blocks and verify the activity
  without repeating the author in the prompt.

Calibrate each route and wording to accept at most 5% of uncredited
coauthors on one publisher. Apply the cutoffs unchanged to the other,
and repeat in reverse. There is no fit to held responses. The 5% cap
is a development criterion and can be exceeded in transfer. A clean
pair accepts the declared contributor and withholds their coauthor.
Measure article-clustered paired intervals and use a warm frozen
Qwen3-Reranker-0.6B. P95 includes selecting and formatting the two
contexts, GPU inference, and serialization. It excludes downloading
and parsing XML and searching an expert inventory.

## Results

Both Slurm tasks of array 45874 completed on ws3 RTX 5090 and
reproduced every non-timing field. One approved Frontiers article
has no contrastive role. Evaluation covers 294 cases across 39
Frontiers articles and 137 across 14 Sensors articles. Each cell
shows accepted credited / accepted uncredited at the frozen
opposite-publisher cutoff.

| Held journal and request | Shared | Length control | Candidate scope | Scoped role only |
| --- | ---: | ---: | ---: | ---: |
| Frontiers, exact | 27 / 5 | 136 / 17 | 291 / 55 | 293 / 72 |
| Frontiers, described | 30 / 15 | 110 / 22 | 179 / 41 | 175 / 69 |
| Sensors, exact | 17 / 9 | 54 / 6 | 44 / 1 | 67 / 1 |
| Sensors, described | 9 / 3 | 16 / 3 | 8 / 3 | 1 / 0 |

For task descriptions, candidate scope raises strict pair ordering from
185 to 255 of 294 Frontiers cases (+23.8 points, article-clustered 95%
interval +17.5 to +30.2) and from 95 to 105 of 137 Sensors cases
(+7.3 points, interval 0.0 to +14.3). The length control orders 191 and
100 pairs, respectively. With explicit role names, scoped ordering is
288 versus 192 on Frontiers and 128 versus 100 on Sensors. Shared and
scoped query P95 are both about 10 ms. The reduction in block counts
does not yield a material latency gain in these short declarations.

A retrospective comparison lets every route retain the most credited
authors without exceeding the corresponding shared route's observed
uncredited count. Pass whole tied score bands together. With task
descriptions, shared versus candidate scope accepts 30 versus 126 at
the 15-uncredited cap on Frontiers and 9 versus 18 at the three-uncredited
cap on Sensors. These are held-label diagnostics. The frozen transfer
cutoffs do not recover those operating points. On Frontiers, scope at
its frozen cutoff accepts 41 uncredited coauthors, above both the
shared route and the 5% development cap. In reverse, it gives up one
credited result at the same uncredited count.

The paired ranking effect transfers most clearly with explicit role
names. These are publisher-declaration labels from resolvable records,
with no claims about contribution quality, identity resolution at
large, or full retrieval. [Live results](results.json) ·
[Matched-cap and ordering analyses](risk-analysis.json).

## Reproduce

Restore the same XML projections with the [source audit](source-audit.json)
and rerun using the pinned open-weight reranker. The release contains
IDs, checksums, model evaluations and author ordinals. It omits
contribution paragraphs and byline names. A fresh EFetch restore
reproduced both the panel and the audit byte for byte.

```sh
PYTHONPATH=src:. python -m experiments.credit.engineering_contribution_panel NEWSNAPSHOT \
  --restore benchmarks/engineering-contribution-transfer-v1/source-audit.json \
  --output PANEL --audit NEW_AUDIT

# Inside a Slurm CUDA allocation. Repeat with --offset 1.
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.engineering_contribution_verify \
  PANEL QWEN_RERANKER NEW_RESULTS --offset 0

PYTHONPATH=src:. python -m experiments.credit.engineering_contribution_analysis \
  NEW_RESULTS NEW_RISK_ANALYSIS
```
