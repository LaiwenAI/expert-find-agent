# Learn which evidence check can improve reviewer ranking

Use the full CoF reviewer inventory and the [fixed centroid–best-source blend](../cof-global-profile-pooling-v1/README.md). Each query retrieves ten reviewers, with up to three papers per reviewer. Discard exact owned-query overlap using the shared input loader. Use its surviving SIGIR queries and the same KDD and SciRepEval panels as the fixed profile comparison. The released SIGIR judgments cover every reviewer. KDD and SciRepEval judgments are partial. The selected panels have no normalized query title in common.

Score all distinct query–source pairs offline with the same frozen Qwen3-Reranker-0.6B, fixed batch cap and truncation rule. A nonnegative margin confirms a source. In replay, all policies read the first source of every candidate in common and see those margins. A source shared by reviewers counts as one read and its margin is visible to all of its owners. Every policy gets five more distinct reads or stops at pool exhaustion. After the budget, place reviewers with two confirmed sources first and fill remaining Top-5 slots in original rank order. Reviewer-level judgments score the ranking. The citation outcome records whether a judged reviewer has two model-confirmed sources. The citations have no independent human support labels on CoF.

| Policy | Follow-up choice |
| --- | --- |
| Rank first | Earliest reviewer in the fixed ranking, then earliest owned source. |
| Round robin | Earliest source position, then earliest reviewer. |
| One hit | Prioritize an unconfirmed reviewer with one observed positive source, then review rank and stable source order. |
| Margin | Prioritize one-hit reviewers, then use the best observed verifier margin and the fixed ordering. |
| Fitted value | Rank eligible reads by expected change in the reviewer-level evaluation objective. |
| Hindsight | Choose the immediate best read with access to all hidden margins and judgments. |

Use a fixed hash of the SIGIR query ID to assign the first two thirds to fitting and the remainder to test. Replay the one-hit strategy for five steps on each training query. At every step label each feasible next read with its counterfactual immediate change in Top-5 known positives plus twice the cited-positive count minus the cited-negative count. Fit a ridge regression of that change from 12 features, including source position, author-sharing count, observed support, verification margins and remaining budget. At test time the learned policy uses observed margins and retrieval state alone. Hyperparameters and eligibility rules stay fixed across all panels. Hindsight is a diagnostic trajectory with a greedy one-action lookahead, so it does not upper-bound the best possible five-action sequence. Full-pool verification is a separate reference.

Report matched read counts, known positive and judged negative counts, query-paired bootstrap intervals, and partial-judgment bounds for expert-set changes. The CPU timings cover the decision steps alone. Fixed source scores isolate the reading order and avoid mixing numerical batch variation with decision quality. Equal numbers of reads do not imply equal token use or end-to-end latency.

## RTX 5090 results

Slurm job 42570 fits on 48 fully judged SIGIR queries and tests on 25 other SIGIR queries, 122 KDD queries and 59 SciRepEval queries. The 4,320 training action–state pairs include 74 positive and 277 negative immediate changes. One KDD source pair reaches the fixed 1,024-token limit. All replay policies use its same truncated score.

| Panel | Policy | Top-5 known positives | Cited positives | Cited judged negatives | Cited unjudged |
| --- | --- | ---: | ---: | ---: | ---: |
| SIGIR | Rank first | 21 | 3 | 7 | 0 |
| SIGIR | Round robin | 21 | 4 | 9 | 0 |
| SIGIR | One hit | 21 | 3 | 11 | 0 |
| SIGIR | Margin | 21 | 3 | 9 | 0 |
| SIGIR | Fitted value | 21 | 0 | 0 | 0 |
| SIGIR | Hindsight | 23 | 6 | 0 | 0 |
| KDD | Rank first | 23 | 7 | 35 | 197 |
| KDD | Round robin | 24 | 13 | 41 | 213 |
| KDD | One hit | 23 | 12 | 39 | 226 |
| KDD | Margin | 24 | 13 | 35 | 216 |
| KDD | Fitted value | 23 | 5 | 4 | 58 |
| KDD | Hindsight | 30 | 18 | 4 | 178 |
| SciRepEval | Rank first | 53 | 19 | 14 | 29 |
| SciRepEval | Round robin | 51 | 18 | 15 | 41 |
| SciRepEval | One hit | 51 | 19 | 18 | 51 |
| SciRepEval | Margin | 50 | 21 | 17 | 47 |
| SciRepEval | Fitted value | 51 | 7 | 3 | 8 |
| SciRepEval | Hindsight | 63 | 33 | 0 | 31 |

Mean unique reads are 15.00, 13.14 and 14.31 per query respectively for every replay policy. Full-pool verification requires 30.00, 25.30 and 28.22 reads and returns 6, 14 and 27 cited positives. The fitted policy makes fewer cited recommendations of every label class without improving the known-positive ranking over the one-hit policy on any panel. Its cited-positive P@5 difference relative to one-hit is −1.15 percentage points on KDD (paired 95% interval −2.30 to −0.16) and −4.07 points on SciRepEval (−7.12 to −1.69). Hindsight returns more known positives at the same read count. Its P@5 difference on each transfer panel can still change sign under the partial-judgment bounds.

The routing objective had sparse gains and many negative training actions. A plausible explanation is that the fitted policy learns to avoid completing recommendations. The current experiment does not isolate reward design from feature limitations. Fixed round robin is a low-overhead control to keep. Compare future policy variants at a common cited-coverage requirement and separate the gain from a read from the usefulness of the rank change it triggers. [Aggregate results](results.json)

## Reproduce

Use the [fixed profile study inputs](../cof-global-profile-pooling-v1/README.md), the SIGIR vector snapshot produced by [the same embedding runner](../../experiments/cof_qwen_embedding.py) with `--dataset SIGIR --backend torch`, and the same open checkpoints in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.cof_followup_value QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
