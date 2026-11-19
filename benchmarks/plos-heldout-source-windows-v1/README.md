# Confirm study identity with abstract windows after withholding the clue sentence

Reuse the 134-article [PLOS ONE publisher snapshot](../plos-compound-contributor-search-v1)
and one supported two-role request per target article. Describe the study
using its last suitable noninitial abstract sentence, omitting every fourth
word starting from the second. Remove the original sentence and its exact
repetitions from the same article before building the index or verifier
windows. Apply that removal policy to all index articles. Remaining passages
can still share words and underlying facts with the clue. This is a
controlled query construction, with no independent natural-language
relevance judgments.

Both policies use the same first four BM25 nominations and score them in a
single batch with frozen Qwen3-4B-Instruct-2507. Both show the title and up to
80 abstract words. The fixed policy uses the lead. The directed policy uses
the overlapping window with the greatest IDF-weighted clue-term coverage.
Both nominate the highest-scoring source, then accept it only if its score
exceeds their respective cutoff. Accepted sources expose only contributors
who have both required publisher-declared roles on that source. The role
filter does not change the source decision.

Split targets by fixed DOI hash order, 30 calibration and 30 test. Every
target is used when present and also masked from retrieval, with fixed index
statistics. The cutoff for each policy is the largest score among its
incorrect calibration nominations across the two views. Test targets did not
select the cutoff. The source search index is shared and the article panel
was previously used under different clue and decision protocols. The latency
covers query retrieval, window choice, and model scoring on a warm GPU.
Offline construction, role lookup, whole-system serving and queue latency
are outside that timing.

## Results

Slurm job 43308 completed on an RTX 5090 on ws5. The intended source
was among the first four for 21 of 30 present test targets. Lexical top-one
chose 16 correct sources without an abstention option. Masking the target
gave another source for all 30 lexical top-one queries.

| Policy | Correct top-ranked before cutoff / 30 | Correct sources accepted / 30 | Wrong accepts, present / 30 | Wrong accepts, masked / 30 | Present-view P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lead window | 17 | 2 | 0 | 0 | 59.9 ms |
| Directed window | 18 | 0 | 0 | 0 | 67.4 ms |

Both policies score 120 source pairs per 30-query view. Two supported
contributor links survive the lead policy, and none survive directed
windowing. Directed windowing gives one additional correct first-ranked
source before thresholding, but produces a more confident wrong proposal
during calibration (26.125 versus 22.25 logit-margin cutoff) and clears its
frozen cutoff for no test query. Zero observed wrong accepts is accompanied
by near-total or total abstention. The results favor retaining the cheap
lead as a comparison, and show why source confirmation must be evaluated
jointly for coverage and wrong-source risk. The publisher declarations
check roles on particular studies and do not establish company fit.
[Aggregate results](results.json) · [Related passage retrieval benchmark](https://aclanthology.org/2024.acl-long.236/)

## Reproduce

Use the original pinned publisher XML directories and the existing public
fixture. Run with the frozen open checkpoint in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.heldout_source_windows \
  benchmarks/plos-compound-contributor-search-v1/public-queries.json \
  QWEN_4B NEW_RESULTS XML_DIRECTORY [XML_DIRECTORY ...]
```
