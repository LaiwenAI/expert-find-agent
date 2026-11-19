# Fixed and learned dense profile scoring

The [CMU Gold benchmark](../cmu-bounded-reranking-v1/README.md) compares paper suitability within each individual reviewer. This experiment asks whether learning to combine several owned sources improves the judgments with only one query embedding. It also transfers the learned score to [MERIT Assessor](../merit-evidence-dispatch-v1/README.md), where candidates assess their suitability for reviewing a paper. The external experiment compares candidates for the same target paper when both positive and negative assessed candidates are present.

## Methods

All routes use the same Qwen3-Embedding-0.6B vectors and the same source history. The fixed controls score the best source, the normalized history centroid, the mean cosine of the three best sources, and the equal mixture of best source and centroid. The learned pooling model uses the best score, centroid score, top-three mean, and within-history score standard deviation. Fit the existing L2-regularized logistic head to differences between two papers with unequal expertise grades from the same reviewer. Include both orders of every training pair. There is no cross-encoder at inference.

The relative model also uses the background mean similarity, the best-source margin against the median maximum from a same-sized sample drawn with replacement from the reference score distribution, and the top-three and history mean excesses above the background mean. Normalize the three excess scores by the background standard deviation with a minimum of 0.025. Estimate the median maximum by using the empirical quantile at `0.5^(1/n)` for a history of `n` sources. Only publications attached to training-side histories enter the background, and documents matching a held-out submission by title or ID are excluded.

## CMU cross-validation

Build connected components from reviewer identity and normalized target title. Keep components intact in five folds and balance by the number of observations, without using grades to split. Each observation is scored once by a model trained without its component. On 477 observations from 58 reviewers, the folds contain 94–98 cases each. The loss uses 1,653 unequal-grade paper pairs, weighted by the original grade difference. Descriptive bootstrap intervals resample whole reviewer–title components with the fitted scores held fixed. They do not measure retraining or tuning uncertainty.

| Route | Weighted pairwise loss ↓ | Warm P95 ↓ |
| --- | ---: | ---: |
| Best source | 23.45% | 7.58 ms |
| History centroid | 24.09% | 7.56 ms |
| Mean of best three | 22.35% | 7.57 ms |
| Equal best source and centroid | **22.10%** | 7.58 ms |
| Learned pooling | 22.27% | 7.60 ms |
| Learned relative | 23.55% | 7.72 ms |

The equal blend reduces observed error by 1.35 percentage points against best source. Its paired descriptive 95% interval is −3.20 to +0.74 points. The best-source error reproduces the prior live CMU control. Neither learned head improves on the fixed blend. Learned relative increases observed error despite using the background calibration features. Job 41596 ran on an RTX 5090. Each method performs its own live query encoding and scoring with resident article vectors. The timing excludes service transport, offline encoding and training. [Aggregates](cmu-results.json)

## Transfer to MERIT

Fit both learned heads once on all CMU training cases. Do not use MERIT scores to fit or select a head. Check overlap by normalized target title against the CMU targets and known publication texts, and by exact normalized history-title signature against the CMU profiles. No rows fail these overlap or minimum-history checks. Repeated pairs with the same target and exact history are collapsed when their assessments agree; two conflicting pairs are excluded. The retained dataset has 604 candidate views on 508 titles. Only 42 targets have both label classes, forming 51 directly comparable pairs.

| Route | Within-target pairwise loss ↓ | Top-one on mixed targets ↑ | Global pairwise loss ↓ |
| --- | ---: | ---: | ---: |
| Best source | 17.65% | 33/42 | 21.09% |
| History centroid | **13.73%** | **36/42** | 20.40% |
| Mean of best three | 17.65% | 34/42 | 22.77% |
| Equal best source and centroid | 15.69% | 35/42 | 19.40% |
| Learned pooling | 17.65% | 34/42 | 19.87% |
| Learned relative | 17.65% | 34/42 | **19.09%** |

The learned relative score ranks positives versus negatives more accurately across targets than the other routes in this run. It leaves same-target pairwise loss equal to best source and behind the centroid. Cross-target score discrimination and candidate ranking within a target support different research claims. MERIT only labels the supplied candidate views, and the 51 unequal-label pairs are clustered in 42 targets. Neither dataset has independent citation-level judgments for these outputs. Job 41598 ran on an RTX 5090, with batched cached embeddings and no live latency claims. [Transfer aggregates](merit-results.json)

## Reproduce

Provide the public input releases locally and a verified open-weight checkpoint in a Slurm CUDA allocation. Outputs are aggregate metrics.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.decision.cmu_profile_ranking CMU_GOLD_ROOT EMBEDDING_CHECKPOINT NEW_CMU_RESULTS
python -m experiments.decision.cmu_merit_transfer CMU_GOLD_ROOT MERIT_TEST_JSON EMBEDDING_CHECKPOINT NEW_MERIT_RESULTS
```
