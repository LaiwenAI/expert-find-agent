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

## Acceptance by separation from the strongest rival

Job 43311 reproduces the previous discrete results and thresholds. Reuse
all four scores from each existing verification batch. Compare the
absolute top score, its difference from the second score, and the same
difference when the verifier and lexical retrieval agree on the top source.
The last rule abstains before thresholding when their nominations differ.
None of these comparisons needs another model call.

For each rule and window policy, freeze the least restrictive strict
cutoff at the specified calibration error budget. Wrong proposals from
the present and withheld calibration views both count. Score ties can
leave some of a budget unused. The primary comparison uses a budget of
zero, with budgets of one and three reported as sensitivity checks.

| Window and acceptance rule | Correct accepted / 30 present | Wrong accepted / 30 present | Wrong accepted / 30 withheld | Supported contributor links |
| --- | ---: | ---: | ---: | ---: |
| Lead, absolute | 2 | 0 | 0 | 2 |
| Lead, rival margin | 5 | 0 | 0 | 7 |
| Lead, margin and lexical agreement | 5 | 0 | 0 | 7 |
| Directed, absolute | 0 | 0 | 0 | 0 |
| Directed, rival margin | 7 | 1 | 2 | 9 |
| Directed, margin and lexical agreement | 10 | 0 | 0 | 12 |

Lexical agreement filters confidently wrong nominations in the directed
route. The directed combined rule recovers 10 more correct sources than
its absolute-score control, with a descriptive paired bootstrap interval
of 16.7 to 50.0 percentage points. The improvement from the lead window's
margin alone is less certain (−3.3 to 23.3 points). Each withheld slice
contains 30 targets, so zero observed wrong accepts leaves considerable
uncertainty. [Thresholds and sensitivity results](acceptance-gates.json)

### Frozen transfer to separate articles

Job 43317 applies those exact zero-budget cutoffs on ws5 with the same
checkpoint. The new index comes from the two previously captured
[PLOS source inventories](../plos-source-gate-transfer-v1). Verify both
publisher receipts, exclude DOI, normalized author name and available
ORCID overlaps with the original raw inventory, then require complete
structured contributor declarations. One record is excluded by name.
The resulting index has 52 articles and 44 eligible study clues. The
holdout uses separate articles with constructed queries from the same
publisher and the same query-generation protocol. It measures transfer
across articles rather than across publishers or to natural enterprise
requests.

| Frozen rule | Correct accepted / 44 present | Wrong accepted / 44 present | Wrong accepted / 44 withheld | Supported contributor links | Wrong-source contributor links, present + withheld |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lead, absolute | 4 | 2 | 2 | 6 | 0 |
| Lead, rival margin | 7 | 0 | 2 | 9 | 4 |
| Lead, margin and lexical agreement | 6 | 0 | 1 | 8 | 2 |
| Directed, absolute | 0 | 0 | 0 | 0 | 0 |
| Directed, rival margin | 11 | 1 | 3 | 14 | 8 |
| Directed, margin and lexical agreement | 11 | 0 | 1 | 14 | 2 |

The directed combined rule keeps all 11 correct acceptances from its
margin-only control and removes three of four wrong source acceptances.
The remaining failure attributes two contributor links to an incorrect
source when the intended study is absent. The same rule costs zero
additional GPU calls over directed absolute confirmation. Overall,
the shared lexical shortlist includes 35 of 44 targets and the directed
verifier ranks 33 first before abstention. These counts support a
candidate confirmation rule with limited coverage and residual
missing-source risk. [Frozen transfer results and DOI inventory](frozen-transfer.json)

## Reproduce

Use the original pinned publisher XML directories and the existing public
fixture. Run with the frozen open checkpoint in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.heldout_source_windows \
  benchmarks/plos-compound-contributor-search-v1/public-queries.json \
  QWEN_4B NEW_RESULTS XML_DIRECTORY [XML_DIRECTORY ...]
```


For transfer, use the original four capture folders after `--previous`
and the two published holdout capture folders, with their receipts, after
`--transfer`. The acceptance-gates file was committed before the transfer
run. It contains the calibration cutoffs and in-domain diagnostic results.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.transfer_source_acceptance \
  benchmarks/plos-heldout-source-windows-v1/acceptance-gates.json QWEN_4B NEW_RESULTS \
  --previous ORIGINAL_XML_DIRECTORY [ORIGINAL_XML_DIRECTORY ...] \
  --transfer TRANSFER_XML_DIRECTORY [TRANSFER_XML_DIRECTORY ...]
```
