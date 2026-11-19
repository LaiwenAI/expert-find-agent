# Retrieve author-role evidence from long publisher articles

Use the older and author-disjoint transfer cohorts of the
[clause rescue benchmark](../plos-anchored-clause-rescue-v1).
For each paper, fix a pair of authors. Both have a shared declared
contribution, and only one has the requested second contribution.
The task supplies the second activity as a paraphrase and the two
unambiguous publisher initials. The common activity selects contrast
cases, but it is not a query feature. Article identity is given.

Index each original article's JATS body in paragraph-local windows
of at most 96 whitespace-separated words, plus the original
contribution declaration split at its period boundaries. Train no
new model. Use the same live, frozen Qwen3-Reranker-0.6B, a personal
verification instruction, and exactly one selected window per
candidate. Every method opens two candidate windows and scores
two prompt–window pairs per query, even if their texts coincide.
All methods share an inverted lexical index with document-frequency
weighting and length-normalized term frequency.

| Selector | Candidate window |
| --- | --- |
| Lexical | Highest activity score anywhere |
| Owner lexical | Highest activity score with a fixed 0.75 weighted boost for the candidate's initials |
| Section lexical | Highest activity score within the publisher contribution section, using its JATS location |
| Owner postings | Highest activity score within the candidate's owner postings, falling back to the global lexical result if empty |

The owner postings are built by recognizing complete
`activity: INITIALS.` strings anywhere, including the main text.
They need no section ID at lookup. The index uses the publisher's
literal local owner list and does not match the activity against a
hand-labeled role dictionary at retrieval. This study assumes that
the original contribution note is available and that the initials
can be resolved. Section lexical is a markup-aware reference.

Calibrate a decision threshold per selector on the earlier split
and wording. Maximize correctly accepted declarations under a 10%
calibration cap on papers with an accepted, unsupported citation.
Test with a fixed second paraphrase. Count a verified hit only when
the returned original publisher declaration assigns the requested
role to the candidate. Material outside that declaration may describe
the research, but this reference does not adjudicate undocumented
roles there. Count accepted partial-match authors and other
unsupported attachments as adverse under this reference. Report
paired intervals by paper in the first test and by author clusters
in the transfer set. Record window disagreement over the entire
panel without selecting examples after observing correctness.
Warm replay P95 includes window selection and live inference from
the in-memory article index. It excludes fetching articles, source
discovery, candidate generation, index construction and queueing.

## Results

Slurm job 43097 ran on one RTX 5090 on ws5. The 30 calibration
papers supply the thresholds. The held-out and transfer articles
contain a median of 80 and 86 indexed windows and about 5,000
words. Each method does 40 and 90 model comparisons, respectively.

| View | Selector | Changed cases vs lexical | Correct clause retrieved | Clean supported papers | Papers with an unsupported attachment | P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Earlier test (20) | Lexical | 0 | 5 | 2 | 0 | 8.98 |
| | Owner lexical | 4 | 7 | 3 | 0 | 8.96 |
| | Section lexical | 9 | 11 | 11 | 3 | 8.76 |
| | Owner postings | 18 | 13 | 6 | 0 | 8.93 |
| Transfer (45) | Lexical | 0 | 12 | 3 | 1 | 8.89 |
| | Owner lexical | 7 | 16 | 5 | 1 | 8.84 |
| | Section lexical | 24 | 24 | 20 | 10 | 8.78 |
| | Owner postings | 41 | 28 | 7 | 0 | 8.79 |

Section routing recovers many clean positives, but its frozen
threshold allows ten adverse papers on transfer. Author postings
find four more correct clauses than the section route there,
while the calibrated verifier accepts only seven of its 28
correct selected clauses. It produces zero observed adverse papers
in these 45 cases. The increase in clean cases relative to global
lexical search has a clustered 95% interval of 0 to 20 percentage
points on transfer. The sample cannot establish a low population
error rate from zero events.

The transfer window word totals are 3,936 for global lexical and
582 for author postings. Component P95 is similar. The evidence
supports an indexing benefit before verification on publisher
syntax and motivates measuring role semantics after a structural
author check. It does not transfer by itself to unstructured
enterprise evidence or prove an end-to-end search speedup.

[Aggregate results](results.json)

## Reproduce

Restore the pinned publisher snapshots using the receipts in the
[independent cascade benchmark](../plos-independent-cascade-v1).
Use the pinned reranker through a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.full_article_role_windows \
    OLD_PUBLISHER_SNAPSHOT TRANSFER_PUBLISHER_SNAPSHOT RERANKER_CHECKPOINT NEW_RESULTS.json
```
