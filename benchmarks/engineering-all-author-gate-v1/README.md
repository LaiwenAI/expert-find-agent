# Normalize one request before checking all author declarations

Reuse the 54 exact-agreement articles from the
[engineering publisher transfer panel](../engineering-contribution-transfer-v1).
For each article issue one request for each of the 14 CRediT roles
and evaluate every author in the paper. The 756 requests include
roles absent from the declaration and roles shared by all authors.
There is no forced one-winner assumption and no annotation-based
choice of rival. Use the original task descriptions and a separate
authored paraphrase bank. Both banks repeat across papers, and
neither represents open-ended technical requirements.

## Decision policies

- **Shared / scoped:** use the frozen Qwen3-Reranker-0.6B to score each
  candidate against the full visible declaration or against just the
  blocks resolved to that author.
- **Centered scope:** use the same model calls and subtract the
  median author score in that article and request. The extra CPU
  reduction is not separately timed.
- **BM25 glossary / embedding glossary:** score each of the 14 role
  definitions once per request. For each candidate subtract the highest
  score of a role absent from that candidate's declaration from the
  highest score of an assigned role. When the difference exceeds a
  nonnegative threshold, even the best competing role hypothesis
  outside the candidate's assigned set is below the acceptance band.
  The visible parser produces role membership without reading the JATS
  reference labels. BM25 uses compiled glossary postings. Dense
  uses frozen Qwen3-Embedding-0.6B with cached definition vectors,
  and encodes the request live. Neither glossary policy uses a reader.

Use the independently versioned [CRediT glossary](https://credit.niso.org/contributor-roles/).
Freeze one cutoff per policy on the original wording in the
other publisher. Maximize the number of credited acceptances subject
to at most 5% of requests admitting any uncredited author. Include
requests with no credited author in the calibration denominator.
Tied scores enter together. Reuse each cutoff on both wording banks
of the target publisher. A nonempty clean result includes at least
one credited and no uncredited author. Exact match additionally
requires recovering the whole set of credited authors, and includes
correct empty outputs. Uncredited means absent from both records and
does not establish that the person lacks the skill.

## Results

Both tasks of Slurm array 45879 complete on ws3 RTX 5090.
All live scores and non-timing results reproduce, with no clipped
inputs. Each wording results in 560 Frontiers requests and 196
Sensors requests. Of those, 212 and 50 ask for an uncredited role,
and 54 and nine ask for roles assigned to every author.

The table reports the alternate wording, which is outside the
phrases used to choose the opposite-publisher thresholds.

| Target journal and policy | Clean nonempty requests | Requests with uncredited authors | Exact author sets | P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Frontiers, shared | 52 | 29 | 228 | 23.17–23.23 |
| Frontiers, scoped | 160 | 34 | 289 | 14.45–16.42 |
| Frontiers, centered scope | 52 | 26 | 232 | 14.45–16.42 |
| Frontiers, BM25 glossary | 142 | 88 | 297 | 0.05 |
| Frontiers, embedding glossary | 247 | 29 | 418 | 7.96–8.01 |
| Sensors, shared | 10 | 2 | 59 | 18.39–18.42 |
| Sensors, scoped | 7 | 2 | 55 | 17.34–17.35 |
| Sensors, centered scope | 24 | 10 | 65 | 17.34–17.35 |
| Sensors, BM25 glossary | 63 | 44 | 87 | 0.05 |
| Sensors, embedding glossary | 103 | 19 | 140 | 7.95–7.98 |

On the alternate wording, the embedding glossary adds 87 clean
Frontiers requests and 96 Sensors requests relative to scoped
verification. It also increases the uncredited rate on Sensors
from 2/196 to 19/196. Frontiers ends at 29/560, just above the
5% ceiling. On the original wording, the embedding route yields
298 clean and 17 wrong requests on Frontiers, and 121 clean and
17 wrong requests on Sensors. Median centering accepts none of
the all-shared author sets in either publisher or wording. The
embedding route recovers 37/54 and 6/9 such sets on alternate
wording.

For a retrospective comparison, cap each held route at the smaller
of 5% of requests and the frozen scoped route's uncredited count.
On alternate wording, the matched cap is 28 for Frontiers and two
for Sensors. After this held-label adjustment, the embedding
route attains 235 and 87 clean requests versus 147 and 23 for
the scoped route. The diagnostic shows headroom for calibration.
It does not demonstrate those scores at a deployable threshold.
Requiring both frozen glossary routes to agree yields 146 clean
and 13 wrong requests on Frontiers, and 57 clean and eight wrong
requests on Sensors. This controls observed errors at a substantial
coverage cost on this panel.

Article-clustered intervals and a separate sensitivity calculation
clustered by the 14 repeated role descriptions are in the release.
The benchmark isolates role normalization and citation attachment
inside a resolved author roster with complete declarations. It does
not test identities spanning several sources, substantive competence,
or open expert search. [Replicated live scores](results.json) ·
[Risk and wording analysis](risk-analysis.json).

## Reproduce

Restore the source panel from the source audit in the prior benchmark.
Use the glossary snapshot checked by SHA-256 in the runner and the
pinned open model revisions. Run inside a Slurm CUDA allocation, with
offsets 0 and 1.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.engineering_all_author_gate \
  PANEL NISO_EXAMPLES QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS --offset 0

PYTHONPATH=src:. python -m experiments.credit.engineering_all_author_analysis \
  NEW_RESULTS NEW_RISK_ANALYSIS
```
