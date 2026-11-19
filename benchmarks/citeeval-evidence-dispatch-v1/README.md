# One-step evidence dispatch on human citation edits

Use the [CiteEval](https://aclanthology.org/2025.acl-long.1574/) metric-development and metric-test releases from the [authors' repository](https://github.com/amazon-science/CiteEval). Each response arrives with a fixed candidate passage pool and existing citations. The task is to select at most one additional passage for a particular statement.

Keep a positive case when at least two annotations assign the statement to retrieval context and agree on the same evidence addition. Keep a no-add control only when three annotators assign the statement to retrieval context and none requests an evidence addition. Treat other judgments as unresolved for this analysis. Independently verify statement text, passage IDs and original citation IDs. Skip consensus edits that point outside the candidate pool or at an already-cited passage. Exclude every test response whose normalized query appears in the development release. The dropped overlap reflects repeated variants within the source release. Cluster paired intervals by normalized query.

| Route | Selection |
| --- | --- |
| Query lexical | Weighted query-term overlap. |
| Claim lexical | Weighted statement-term overlap. |
| Residual lexical | Weighted overlap for statement terms absent from the currently cited passages. |
| Neural top two | Run a frozen Qwen3-Reranker-0.6B on the two uncited passages ranked highest by residual lexical overlap. |
| Neural all | Run the same checkpoint and statement prompt on every remaining passage. |

Lexical IDF is computed from the supplied response pool and uses compiled term sets. Neural routes are charged for the passage texts supplied to the model. Each route selects only one additional citation or abstains. Set a separate threshold for each route that accepts at most 10% of development no-add cases. Report forced-choice hits, gated hits, nonconsensus selections on positive cases, and acceptance on held-out no-add controls. These are agreement measurements against observed edits. A different passage may also be appropriate without appearing in the consensus. No-add acceptance is a resource diagnostic and does not establish factual harm.

Warm P95 includes lexical selection and the stated verifier pass. It excludes offline term compilation, full-corpus document retrieval and service overhead. This is a statement-evidence task with human annotations. It does not identify the personal contributions of an expert.

## RTX 5090 results

Slurm jobs 42707 and 42712 ran on ws5 and reproduced all decision counts. The development release has 154 normalized queries, all repeated somewhere in the test release. Removing the shared queries excludes 416 test responses. The resulting evaluation has 791 eligible statements in 138 normalized query groups. Of these, 89 have a consensus addition and 702 are no-add controls. The calibration sample has 32 positives and 248 no-add controls. Upstream consensus points to an existing citation or an unavailable passage in 7 development and 32 retained test statements, which are excluded.

| Route | Forced hits / 89 | Gated hits / 89 | Gated on no-add / 702 | Mean neural passage inputs | Warm P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query lexical | 17 | 0 | 0 | 0.00 | 0.041 ms |
| Claim lexical | 63 | 31 | 64 | 0.00 | 0.034 ms |
| Residual lexical | 61 | 0 | 0 | 0.00 | 0.024 ms |
| Neural top two | 68 | 47 | 109 | 2.00 | 9.053 ms |
| Neural all | 68 | 43 | 116 | 8.39 | 32.787 ms |

The two-passages shortlist contains an annotated addition in 74 of 89 positive statements. Neural reranking with two passages and with the full uncited pool each selects an annotated passage 68 times. The paired 95% interval for the forced-choice difference is −10.19 to +10.26 percentage points. The bounded route reduces P95 by 72.4% in this component. The thresholded comparison favors two passages by four statements, with a paired interval of −4.94 to +13.10 points. The realized no-add acceptances are 15.53% and 16.52%, so the nominal 10% calibration rate does not transfer. Claim lexical accepts 9.12% of no-add cases and remains a strong low-cost control.

Seventy of the 89 positive statements initially have no citation. Among the 19 already cited statements requiring more evidence, the bounded and full paths find 14 and 15 in forced choice, or 5 and 7 after gating. The aggregate result is driven largely by locating a first citation. A stronger test of adaptive expert search still needs more queries with partially satisfied constraints, text proving the nominee's contribution and an independent end-to-end evaluation. [Aggregate results](results.json)

## Reproduce

Obtain the metric-release files from the authors' download, and save them as `dev.metric`, `dev.human.out`, `test.metric`, and `test.human.out`. The script pins all four hashes. The upstream dataset is used for research and is not redistributed here.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.citeeval.evidence_dispatch CITEEVAL_DIRECTORY QWEN_RERANKER NEW_RESULTS
```
