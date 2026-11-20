# Test verification routing at a common adverse-issuance budget on RELISH

Validate the [request-level routing rule](../csfcube-facet-fallback-v1)
on a separate scientific relevance collection. [RELISH](https://academic.oup.com/database/article/doi/10.1093/database/baz085/5608006)
contains scientist-judged document similarity. The
[RELISH-Aspire release](https://figshare.com/articles/dataset/RELISH-Aspire/19425506)
provides the titles, sentence-segmented abstracts, official seed splits
and adjudicated relevance grades. Grade 0 is nonrelevant, grade 1 is
partially relevant and grade 2 is strongly relevant.

Freeze 65 fit requests and 55 risk-calibration requests from the
official development split and 100 held requests from the official
test split. For every seed, use a fixed hash sample of 24 judged
candidate papers. Selection ignores the relevance grade. Require
a title and abstract, and exclude article IDs or normalized exact
titles that appear elsewhere in the sample. The complete panel has
5,500 unique articles including the seeds. The released manifest
contains document IDs and official grades. It does not redistribute
abstract text.

## Online protocol

One query embedding ranks the 24 judged candidates by dense similarity
to whole abstracts. Take the first eight. Use that same vector with
cached compact candidate views. RELISH has no provided rhetorical
facet, so the compact view is the title and first three abstract
sentences. Compare:

- Four fixed dense reads, or four reads selected by compact scores.
- Compact cosine alone without model verification.
- A whole-request fallback assigned by stable hash, nearest cosine
  threshold margin, or the ridge-predicted benefit of verifying.

All routes issue at most two results in dense order. The verifying
branches use the same compact query and evidence text. Fit candidate
cutoffs and the five-feature router on the fit group. Use the same
request utility as CSFCube and target 30% fallback on fit. The protocol
refits the learned rule for the new task. It does not claim zero-shot
transfer of CSFCube weights.

For each arm, use the independent risk-calibration set to choose
the smallest upward adjustment to its normalized decision margin
such that at most 10% of requests emit any grade-0 document. The
normalization uses fixed fit-set score scales. Apply those cutoffs
unchanged on the held group. After evaluation, also report a
matched-adverse diagnostic that removes already issued results
from the lowest margin up until no arm exceeds the smaller
adverse-query count of fixed dense four and learned fallback.
It never replaces a removed result and does not tune the
deployment cutoff. Ties can produce a lower realized adverse count.
Strong-return intervals use a paired bootstrap by seed paper.

Measure the warm local P50 and P95 through encoding, ranking,
verification when routed, and serialization. Index preparation, remote
source fetching and service load are outside this timing. Grade labels
evaluate paper-level similarity without assessing author contribution.

## Results

Both ws3 RTX 5090 tasks of Slurm array `45771` complete, each with
two rotated execution orders. The four passes reproduce all
non-timing results. The held panel has 100 judged requests. All arms already meet the
10% adverse-query target on the separate risk-calibration group
(two to four adverse requests out of 55), so no extra shift is
applied. The table reports results before trimming unless noted.

| Route | Strong-return queries | Adverse queries | Grade-1 citations | Strong queries after matching adverse cap | Requests verified |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense four | 49 | 2 | 12 | 49 | 100 |
| Compact four | 48 | 0 | 13 | 48 | 100 |
| Cosine | 62 | 3 | 20 | 60 | 0 |
| Hash routing | 61 | 2 | 14 | 61 | 31 |
| Margin routing | 54 | 1 | 15 | 54 | 23 |
| Learned routing | 65 | 2 | 22 | 65 | 32 |

The learned route gains 16 percentage points in strong-return
queries versus dense four at the same realized grade-0 risk
(paired 95% interval +8 to +25 points). The negative count is
small, and partially relevant citations increase from 12 to 22.
The gain is smaller against the vector and hash controls. Pure
cosine retains 60 strong-return queries after requiring at most
two adverse requests. Learned routing lowers warm request P50 from
26.68–26.77 to 8.56–8.62 ms, and P95 from 31.33–31.48 to
28.16–28.40 ms. Pure cosine has a P95 of 8.60–8.68 ms. These are
paper-relevance results.
For a qualification requiring direct support, partial relevance
needs its own risk endpoint.

For a stricter definition that also counts grade-1 issuance, see the
[fixed-risk follow-up](../relish-strict-risk-v1).

[Aggregate results](results.json) · [Frozen membership](relish-panel-manifest.json).

## Reproduce

Download the three files from the pinned RELISH-Aspire version 1.
Build the panel, then run on a Slurm compute node with a CUDA allocation.

```sh
PYTHONPATH=src:. python -m experiments.csfcube.relish_risk_panel \
  RELISH_SOURCE_DIR NEW_PANEL NEW_MANIFEST

PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.csfcube.relish_risk_routing \
  PANEL MANIFEST QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
