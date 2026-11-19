# Exact headroom for evidence scheduling and reviewer ranking

Keep the four CoF query panels, ten candidate reviewers, three sources per candidate, frozen verifier and first-source wave from the [factorized follow-up experiment](../cof-factorized-followup-v1/README.md). Score each distinct query–source pair once. A source counts for every reviewer who owns it in their fixed prefix. A reviewer is verified after at least two nonnegative margins. The reference is the one-hit policy with five follow-up reads, or pool exhaustion.

Enumerate every feasible subset of up to five further model-positive reads. Negative reads cannot affect the confirmed set or consume fewer reads, so omitting them preserves every reachable output under an at-most-five cap. Store each reachable confirmation mask with its minimum number of reads. The enumeration has access to test labels and to the verifier outputs of every candidate source. It describes best-case headroom and is not a serving method.

| Layer | What can change |
| --- | --- |
| Baseline | One-hit scheduler and the original rank-first order among confirmed reviewers. |
| Exact source schedule | Any reachable evidence state within the same cap. Keep the original ranking rule, the exact number of citations for that request and at most the baseline count of cited judged negatives. |
| Free ranking | Keep the same read cap, citation count and judged-negative cap. Choose any verified reviewer to fill each cited slot. Reject states with too few verified reviewers to fill the slots within the judged-negative cap. |
| Full source pool | Keep the citation count and find all candidates with two accepted sources in their original three-source prefix. |
| Candidate shortlist | Keep the citation count and look up the number of judged positives among the ten candidates, without the evidence constraint. |
| Released labels | Keep the citation count and count all released positives for the query, without the shortlist constraint. |

The full-source state contains all reviewers verified within the budget; any additional known positive can replace an already eligible reviewer without increasing the judged-negative count. Later layers progressively remove evidence and recall constraints. A separate unrestricted calculation drops the baseline citation-count limit to reveal headroom even when the baseline emits no citations. On partially judged panels, known-positive gains do not bound full-population gains, and the judged-negative cap applies only to released labels. CoF does not include human adjudication of the returned citations.

The runner checks query and label hashes and reproduces the prior reference counts on all four panels. A synthetic comparison matches the positive-only enumeration and minimum read costs against full enumeration on 150 shared-source cases.

## RTX 5090 results

Slurm job 42586 reproduces the reference on all 240 queries. Each column counts known positive reviewer–query slots. The citation count is held fixed separately for each query.

| Panel | One hit | Exact schedule | Free ranking | Read all shortlisted sources | Know all shortlist labels | Include all released positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SIGIR (25) | 3 | 5 | 5 | 5 | 7 | 14 |
| KDD (122) | 12 | 18 | 18 | 18 | 41 | 156 |
| SciRepEval (59) | 19 | 30 | 30 | 31 | 46 | 58 |
| NIPS (34) | 23 | 30 | 30 | 31 | 40 | 46 |

The exact schedule can recover 26 known positives on 23 queries without increasing the per-query cited-judged-negative count. Free ranking adds none. Exhausting the current three-source prefix adds two. Removing the evidence constraint adds 49, and also removing the ten-reviewer restriction adds 140. Those last two gaps describe available labels. They do not establish that a new retrieval or evidence verifier can recover the missing experts.

Relaxing the matched citation count, the five-read cap exposes 6, 18, 35 and 30 verified known positives, versus 6, 18, 36 and 31 when scoring the complete candidate-source pool. The exact computation examines 36,788 subsets of additional model-positive reads. The main bottleneck in the current closed pool is the supply of relevant, model-verifiable candidates, with material but smaller headroom from better scheduling. Prioritize candidate and source selection at unchanged verification depth. To distinguish insufficient sources from verifier mistakes, label a diverse set of source excerpts for missed known positives. [Aggregate results](results.json)

## Reproduce

Use the [same public CoF data and models](../cof-factorized-followup-v1/README.md) and run in a Slurm GPU allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.cof_evidence_headroom \
  QWEN_EMBEDDING QWEN_RERANKER FACTORIZED_RESULTS NEW_RESULTS
```
