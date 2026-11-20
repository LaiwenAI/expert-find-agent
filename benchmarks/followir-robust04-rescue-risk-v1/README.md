# Transfer real-candidate rescue cutoffs to FollowIR Robust04

The two previous FollowIR collections showed that adjusting a low-cost
reranker threshold can recover constraint-matching rescue candidates
while also accepting candidates invalidated by the instruction change.
A [synthetic-only calibration](../followir-rescue-contrast-head-v1)
accepted nearly every judged negative. This study calibrates on real
Core17 and News21 candidates and transfers once to Robust04.

The [pinned Robust04 release](https://huggingface.co/datasets/jhu-clsp/robust04-instructions-mteb/tree/0a3efedfcac0a7f859c46cff3a0fac0f8747b28f)
contains 52 paired topics and 722 changed document judgments.
Exclude topics that overlap either calibration collection at 75% query
token Jaccard or 70% instruction five-token shingle Jaccard.
Two topics are excluded before holdout scoring. Candidate pools,
model revisions, and upstream file checksums are frozen.

Use four topical candidates and, on abstention, two remaining candidates
ranked with the revised instruction. The initial threshold is determined
from judged negatives in the pooled topical top six on Core17 and News21.
For each calibration topic, exclude that topic from initial threshold
selection before producing its rescue candidates. On holdout, use the
pooled initial threshold. The initial threshold is fixed at the empirical
10% cap for every rescue policy.

Keep the same candidates and Qwen3-Reranker-0.6B margins for all
comparisons. Fit rescue acceptance thresholds under 5%, 10%, and 20%
empirical limits on known negative acceptances:

- Topical negatives or real rescue negatives from Core17, News21
  or their pool.
- The lower-risk of the two collection-specific thresholds, expressed
  as the larger required score.
- The same worst-collection threshold with an additional budget on
  pooled invalidated documents. These are topically plausible
  documents whose relevance changes from positive to zero.

Count accepted changed positives, known negatives, invalidated
documents and unjudged candidates. Invalidated acceptances are a
subset of known negatives. A retrospective score ceiling also
applies both risk caps directly on the held-out labels. It diagnoses
whether threshold selection or ranking is limiting and cannot be
used as a deployable policy. Thresholds themselves add no
inference calls. This test concerns document relevance, not expert
attribution or calibrated enterprise risk.

## Results

Slurm job `43648` ran on one RTX 5090. It encoded 47,492
distinct documents in 297.5 seconds before the online replay.
The pooled first-pass cutoff is 4.375, from 192 topical negatives.
Cross-fitted calibration supplies 34 Core17 and 44 News21 rescue
candidates. Robust04 triggers rescue on 45 of 50 eligible topics
and proposes 90 documents. These include 33 positives, 49 known
negatives (18 judged invalidated), and eight unjudged documents.

| Rescue threshold at 10% development caps | Cutoff | Positive / 33 | Negative / 49 | Invalidated / 18 | Unjudged / 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial topical cutoff | 4.375 | 1 | 0 | 0 | 0 |
| Core17 rescue | 1.375 | 8 | 4 | 3 | 0 |
| News21 rescue | 2.500 | 3 | 3 | 2 | 0 |
| Pooled rescue | 2.375 | 3 | 3 | 2 | 0 |
| Worst collection | 2.500 | 3 | 3 | 2 | 0 |
| Worst collection plus invalidated cap | 2.500 | 3 | 3 | 2 | 0 |
| Retrospective Robust04 ceiling | 3.3125 | 1 | 2 | 1 | 0 |

The pooled and worst-collection cutoffs satisfy an aggregate
10% negative cap on Robust04 (3/49), but accept 2/18
invalidated documents (11.1%). The extra development constraint
does not change the cutoff at the 10% operating point. The
Core17-only cutoff recovers eight positives and accepts 3/18
invalidated negatives. Even a scalar threshold selected with the
Robust04 labels recovers just one positive under both observed
10% caps. The retrospective ceiling is eight positives under
20% caps. It is a diagnostic of the fixed margin, not a risk
guarantee or a ceiling for other verifiers.

The benchmark separates positive proposals from safe acceptance.
It suggests testing verification with different evidence and
direct supervision for constraint violations. Broader retrieval
or a lower acceptance threshold alone leaves the score separation
problem unresolved. Results are descriptive for these fixed
document pools and do not measure expert relevance or contribution.

[Aggregate results](results.json)

## Reproduce

Download the six parquet files from the pinned publisher release
using the filenames in the repository tree. Name them
`corpus.parquet`, `default.parquet`, `instruction.parquet`,
`qrel_diff.parquet`, `queries.parquet`, and `top_ranked.parquet`.
The preparation command verifies all six digests.
Prepare Core17 and News21 following the
[original rescue study](../followir-constraint-rescue-scope-v1).
Run the cache and evaluation in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. python -m experiments.followir.robust04_rescue_risk prepare \
  ROBUST04_PARQUET ROBUST04_JSON
PYTHONPATH=src:. python -m experiments.followir.decision_features cache \
  ROBUST04_JSON QWEN_EMBEDDING ROBUST04_VECTORS
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.followir.robust04_rescue_risk run \
    CORE17_JSON CORE17_VECTORS NEWS21_JSON NEWS21_VECTORS \
    ROBUST04_JSON ROBUST04_VECTORS QWEN_EMBEDDING QWEN_RERANKER \
    benchmarks/followir-constraint-rescue-scope-v1/results.json NEW_RESULTS.json
```
