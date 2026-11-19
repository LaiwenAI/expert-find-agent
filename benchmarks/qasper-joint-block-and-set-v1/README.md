# Combine next-block routing and evidence-set selection

Use the same official train, development and test paper splits as the [next-block study](../qasper-next-evidence-block-v1/README.md) and the [two-swap evidence selector](../qasper-set-repair-v1/README.md). Preserve each split's question IDs, frozen model checkpoints, dense ranking and first eight verified passages. Offer one further block from dense ranks 9–32 and return at most five paragraphs. Full blocks contain eight paragraphs and all routes obey the same 16-paragraph maximum verification budget.

Cross three reading policies with two final selectors. Compare the fixed next dense block, IDF-weighted lexical gap and a fitted reading preference. The final selector is either pointwise ranking or the existing joint two-swap model. Fit the joint model on fixed-block training outputs and tune its threshold on fixed-block development outputs. Then fit the reading preference separately for each final selector. Its training objective is a complete annotated evidence set in the final response, using preferences between alternative blocks from the same question. Freeze the reading switch threshold on development. Do not pass annotations, reward or alternate block verdicts to the read policy. Ground-truth sets can differ across annotators. Any fully contained set counts as success.

Replay every block once for counterfactual analysis using the fixed eight-passage batch shapes. Record the best possible block for each frozen final selector. This hindsight ceiling measures the room for block routing given that selector and those actions. It does not bound performance of a different output selector. Record the less restrictive shortlist ceiling, which counts evidence available in at least one read pool. Reproduce previously calibrated thresholds and fixed-route results before interpreting the crossing.

Timing is factored from measured query encoding, first-block verification, the selected second block, routing and final selection. Fixed read ordering is not charged for routing features. Pointwise selection is not charged for joint features. No timings include storage, initial paper discovery or network service. QASPER annotates paragraph evidence inside a given paper and does not label expert attribution.

## RTX 5090 results

Slurm job 42501 reproduces the fixed-route thresholds and pointwise and joint evidence results. Fit on 370 papers, calibrate on 112, and score 137 test questions.

| Final selector | Fixed second block | Lexical-gap second block | Calibrated learned block | Best-block hindsight ceiling |
| --- | ---: | ---: | ---: | ---: |
| Pointwise | 43 | 40 | 41 | 44 |
| Joint two-swap | 51 | 49 | 51 | 57 |

All entries count complete returned evidence sets. The joint selector recovers eight more complete sets than pointwise ranking when both use the same fixed reads. It replaces at least one final paragraph for 111 requests. Fixed-route P95 rises from 74.72 to 75.21 ms in the factored measurement. The lexical-gap route under the joint selector gains one complete result and loses three relative to its fixed counterpart. Its paired 95% interval includes zero.

The joint selector increases the number of test questions whose output quality depends on the chosen block from eight to 17. The hindsight ceiling rises from 44 to 57 complete responses, while the ceiling for getting a complete set into the shortlist remains 96 in either case. The fitted block router with the joint selector cannot beat the fixed route on development, and its chosen rule keeps all fixed reads in test. Thus the observed gain comes from joint final selection. Of the six remaining output gains possible under that frozen selector, none was obtained by the learned routing policy. This experiment reuses a known-paper, multi-evidence test panel and does not demonstrate expert-ranking gains. [Aggregate results](results.json)

## Reproduce

Use the [pinned QASPER inputs](../qasper-next-evidence-block-v1/README.md#reproduce), both frozen checkpoints and the two public reference result files. Run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.qasper.joint_block_and_set \
    QASPER_DIR QASPER_TRAIN_DEV_ARCHIVE QWEN_EMBEDDING QWEN_RERANKER \
    NEXT_BLOCK_RESULTS SET_REPAIR_RESULTS NEW_RESULTS
```
