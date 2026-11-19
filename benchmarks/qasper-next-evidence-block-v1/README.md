# Learn whether a second evidence block will complete the result

Use the [QASPER evidence-gap audit](../qasper-evidence-gap-audit-v1/README.md) and its one-question-per-paper multievidence sampling policy. Fit on the official paper-disjoint training split, select the switch threshold on development, and evaluate on the earlier 137-question test panel. Each question has at least one annotator evidence set that can fit in the five output slots. The given paper is already located. Questions, paragraph text and model inputs are unchanged.

All methods rank paragraphs with the frozen embedding model and verify the first eight. Partition dense ranks 9–32 into up to three fixed eight-paragraph blocks. Each policy opens at most one more block. It ranks the paragraphs from the first and second reads by the same combination of dense score and live cross-encoder score, then returns the best five. A short final block can use fewer than eight checks.

The fixed control opens the next dense block. The lexical-gap control opens the block containing the most IDF-weighted question terms absent from the three strongest verified anchors. The learned route estimates the relative value of each action from the block's dense strength, lexical coverage and novelty, paragraph-to-anchor embedding similarity, and section location. Fit a ridge preference on the differences between complete and incomplete actions on the same training request. A complete action returns every paragraph in at least one annotator evidence set. Choose when to override the fixed action using development evidence and favor fewer overrides for tied quality. Keep a forced learned variant as an uncalibrated ablation.

All eligible actions are separately scored for the offline counterfactual evaluation, using their original verifier batch. The factored cost of a policy includes shared query, first-block verification, feature construction, and the measured call on the selected block. Timing excludes indexing, network and storage. Gold sets are used for training and evaluation of the actions, and never passed to a deployed selector. The hindsight ceilings inspect annotated outcomes of all actions and are diagnostic only.

## RTX 5090 results

Job 42492 fits on 370 papers and calibrates on 112. Only 22 training questions and one development question have alternate blocks that change whether the final evidence is complete. The learner fits 43 ordered preference comparisons. Development chooses a zero-margin switch, improving one question over the 33-question fixed result while switching on 44 requests.

| Second read | Complete shortlisted evidence / 137 | Complete final evidence / 137 | Gains / losses vs fixed | Mean reads | Factored P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed next eight | 83 | 43 | 0 / 0 | 15.98 | 74.79 ms |
| Lexical gap | 79 | 40 | 0 / 3 | 15.89 | 75.37 ms |
| Learned | 76 | 41 | 0 / 2 | 15.92 | 75.11 ms |

The fixed route reproduces the earlier flat-16 evidence counts. Even a hindsight choice of the best second block would increase complete final evidence from 43 to at most 44, although it can make a complete set available in the shortlist for 96 requests. Under this specific block layout, fixed verifier and pointwise final selector, better routing has little room to help the output. The learned route's paired change is −1.46 percentage points with a descriptive interval of −3.65 to 0.00 points.

Keep fixed 16 as the equal-budget control. Test the read decision together with a set-aware final selector before expanding the modeling of next-block choice. These timings use the same factored feature-building prefix for every route, even though the standalone fixed route could skip it. The ceiling is for one block among the specified dense ranges and cannot bound other reading actions or other output selectors. [Aggregate results](results.json)

## Reproduce

Use the pinned [QASPER preparation and checkpoints](../qasper-set-repair-v1/README.md#reproduce). Execute in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.qasper.next_evidence_block \
    QASPER_DIRECTORY QASPER_TRAIN_DEV_ARCHIVE QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
