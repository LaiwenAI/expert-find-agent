# Separate evidence completion from reviewer relevance

This experiment follows the [equal-read CoF study](../cof-followup-value-v1/README.md). It keeps the dataset split, ten-reviewer candidate pool, top three owned papers, frozen model verdicts and budget of at most five distinct reads after the first-source wave. It adds the surviving NIPS queries as a fourth evaluation panel. The runner verifies that the four panels share no normalized query titles. It also verifies the original policy outcomes on the previous three panels. The replay scores any shared owned source once.

Use the same 48 fully judged SIGIR training queries. Collect source and reviewer features from the one-hit policy trajectory. Fit one regularized logistic model to predict whether a candidate source will receive a nonnegative reranker margin and another to predict whether the reviewer is relevant under the released hard judgments. Both estimators use only the retrieval state and verdicts already observed at the given step. Only the training targets use a pending source's verifier score or the reviewer's label.

At each decision, simulate the effect on the output if the proposed source were accepted. If any possible next read can increase the number of cited Top-5 slots, consider the actions with the highest such increase. Otherwise follow the one-hit policy. The completion policy chooses the largest estimated probability of verifier acceptance. The factorized policy multiplies that probability by the expected change in relevant Top-5 reviewers plus the expected change in relevant cited reviewers. Reviewer relevance estimates remain fixed during a single decision. Outputs are then scored with the selected actual verdict, and no method uses test judgments to choose a read.

Compare both policies with round robin, one-hit, margin and direct reward regression. In the matched-coverage analysis, take the smaller of two citation counts separately per query. Retain that many of each method's cited reviewers in their original retrieval order. Report both methods' outcomes on the same number of cited slots, with partial-label bounds and known-positive paired intervals. This is a diagnostic and uses the other policy's output to set the common count. Raw results report each policy's complete output. Citation support comes from the reranker, and the public labels assess reviewer relevance. CPU timings measure selection logic only.

## RTX 5090 results

Slurm job 42577 reproduces four prior policy controls on all three original panels. It evaluates 240 queries in total, including 34 additional NIPS queries. The source predictor sees 614 verifier positives among 4,320 training actions. The reviewer predictor sees 390 positive labels among 2,400 training states. Both counts include repeated observations across training states.

| Panel | Policy | Top-5 known positives | Cited positives | Cited judged negatives | Cited unjudged |
| --- | --- | ---: | ---: | ---: | ---: |
| SIGIR (25) | One hit | 21 | 3 | 11 | 0 |
| | Completion | 22 | 5 | 10 | 0 |
| | Factorized | 23 | 6 | 9 | 0 |
| KDD (122) | One hit | 23 | 12 | 39 | 226 |
| | Completion | 23 | 13 | 40 | 237 |
| | Factorized | 21 | 11 | 39 | 236 |
| SciRepEval (59) | One hit | 51 | 19 | 18 | 51 |
| | Completion | 51 | 19 | 20 | 51 |
| | Factorized | 54 | 23 | 20 | 49 |
| NIPS (34) | One hit | 30 | 23 | 25 | 31 |
| | Completion | 30 | 22 | 29 | 30 |
| | Factorized | 31 | 23 | 28 | 28 |

The direct reward model returns 0, 5, 7 and 3 cited positives. Factoring the decision while prioritizing attainable citation coverage restores citation output at the same unique-read budget. The completion-only arm also restores coverage, isolating the marginal role of reviewer relevance.

| Panel | Matched cited slots | Factorized vs. one-hit cited positives | Matched cited judged negatives, factorized vs. one hit | Paired 95% interval for cited-positive P@5 change |
| --- | ---: | ---: | ---: | ---: |
| SIGIR | 12 | 5 vs. 3 | 7 vs. 9 | 0.00 to +4.00 points |
| KDD | 268 | 10 vs. 11 | 37 vs. 38 | −0.49 to 0.00 points |
| SciRepEval | 82 | 19 vs. 18 | 19 vs. 17 | 0.00 to +1.02 points |
| NIPS | 72 | 23 vs. 23 | 24 vs. 23 | −1.76 to +1.76 points |

Against completion-only, the factorized policy changes matched cited-positive counts by +1, −2, +1 and +1. The partial-label bounds span both signs on the three transfer panels. Decision P95 is 0.90–0.98 ms for the factorized policy versus 0.34–0.36 ms for one-hit, excluding retrieval and model inference. All policies average 15.00, 13.14, 14.31 and 15.00 unique reads per panel. The policy removes the strong citation suppression of direct reward learning, but its relevance component does not reliably improve precision at matched coverage. Continue using simple rules as controls. [Aggregate results](results.json)

## Reproduce

Use the inputs from [the original comparison](../cof-followup-value-v1/README.md) and generate the NIPS vectors using [the same embedding runner](../../experiments/cof_qwen_embedding.py) with `--dataset NIPS --backend torch`. Run both models inside a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.cof_factorized_followup \
    QWEN_EMBEDDING QWEN_RERANKER FIRST_COMPARISON_RESULTS NEW_RESULTS
```
