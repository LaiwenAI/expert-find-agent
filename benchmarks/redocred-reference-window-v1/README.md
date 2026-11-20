# Test four-sentence selection against an audited evidence denominator

Reuse the pinned [binding-verifier panel](../redocred-binding-marker-v1)
with two people and two works for the same role. Exclude a development
case shorter than four sentences. The fit, calibration, and held panels
contain 237, 63, and 62 cases. Every route submits the same four selected
original sentences to each of the four relation checks using the same
frozen Qwen3-Reranker-0.6B. The number of words can vary across windows.
Fit independent thresholds on development at a 5% ceiling for accepting
unjudged links. Unjudged links are not known negatives.

## Selection

1. **Global:** greedy coverage of the four names and role cues.
2. **Scoped:** greedy coverage of the four candidate relations, using
   sentence-level person, work, and role similarity.
3. **Learned pair union:** retrain the earlier learned pair-sum selector
   on eligible upstream training documents. Send the union of its four
   selected sentences to every claim, matching the other arms.
4. **Random removal control:** train on four-sentence windows that
   contain all reference sentences, paired with randomly sampled windows
   lacking some reference evidence.
5. **Single reference removal:** use the same positive windows and
   paired training objective. Replace one reference sentence with an
   outside nonreference sentence, keeping the other three sentences.

The last two arms fit the same linear score on maximal pair features
aggregated over the six sentence pairs in a window. They use the
same 1,033 comparisons across 281 labeled training claims and
use the reference annotations only to construct training pairs.
At inference, enumerate candidate four-sentence windows, sum
bounded claim scores, then send one window to the reader. The scores
are ranking utilities, without a probability interpretation. Nearby
work estimates sentence importance or selects evidence for a single
document-level claim. This test compares complete shared windows
under a fixed budget across competing person–work claims.
[SIEF](https://aclanthology.org/2022.naacl-main.212/) ·
[DocInfer](https://aclanthology.org/2022.emnlp-main.51/)

## Results

Both runs of Slurm array 45852 on ws3 RTX 5090 reproduce every
non-timing result and selected sentence ID. The evaluation has
166 known and 82 unjudged links. Only 63 known links supply a unique,
nonempty set of reference sentence IDs. Of the remaining 103,
100 have one record with an empty evidence list, and three lack a
unique reference set. The expanded annotation provides the same
reference lists for these known links. Thus 60/166 is a joint count
of coverage and annotation availability. **The coverage denominator
for observable reference sets is 63.** A gold-aware budget oracle
can cover all 63 by choosing, for each case, the most links whose
reference sets have a union of at most four sentences.

| Selector | Complete reference windows / 63 | Accepted known / 166 | Accepted unjudged / 82 | Accepted with complete references | P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global | 60 | 33 | 4 | 17 | 14.58–14.82 |
| Scoped | 54 | 41 | 4 | 20 | 14.68–14.93 |
| Learned pair union | 60 | 38 | 4 | 21 | 15.19–15.26 |
| Random removal control | 59 | 39 | 4 | 20 | 14.83–15.16 |
| Single reference removal | 59 | 39 | 4 | 20 | 14.77–15.14 |

The learned pair union adds four accepted links with complete
references at the same observed unjudged count. The paired difference
per case is +0.065, with a title-clustered 95% interval [0, +0.176].
Its selector alone takes approximately 1 ms at P95. It covers the
same number of reference sets as global selection. The two removal
controls also tie on aggregate results. Selection alone has little
observable coverage headroom on this subset, and no complete-reference
claim is made for the 103 links without usable evidence labels.
For the global route, the reader accepts only 17 of the 60 known
links whose annotated sentences are present. That gap provides a
separate verification test. Evidence overlap does not establish
entailment or expert suitability.

[Results and selected sentence IDs](results.json).

## Reproduce

Use the same input revisions and reranker as the previous binding
experiment. Run in a Slurm CUDA allocation for offsets 0 and 1.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.docred.reference_preserving_window \
  REDOCRED_DIR RE2DOCRED_DIR QWEN_RERANKER NEW_RESULTS --offset 0
```
