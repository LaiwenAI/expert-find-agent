# Check only sources that can enter a cross-publication answer

Reuse the same local first-query route, joint nomination, two
publisher snapshots, frozen checkpoints, 64 candidates per
condition and three source declaration reads as the
[live benchmark](../live-joint-source-request-v1). The request
supplies topic strings and controlled CRediT roles. Build only
same-ORCID proofs whose two papers satisfy the requested declared
roles. Rank them by the weaker topic confidence, then the sum,
then the original retrieval order.

| Arm | Topic checks after nomination |
| --- | --- |
| Rank | Use reciprocal-rank-based ordering without a reader |
| Full | Score all six condition–source pairs in one 0.6B forward |
| Viable | Score the union of pairs that can actually enter a complete same-person proof, in one forward if nonempty |
| Top pair | Check only the two citations in the best rank-based proof, in one forward if it exists |

The rank control uses the negative ordinal rank as its confidence
score. The neural arms use the frozen yes-minus-no margin. The
first two neural routes maximize the weakest confidence over
the same set of structured proofs. Different physical model
batches can still produce different scores. The top-pair route
fixes the candidate before scoring.

Freeze a confidence cutoff that allows at most 20% issued
answers with a missing required publisher topic tag in the
calibration data and requires at least five issued calibration
answers. Take the most permissive feasible cutoff. Abstain when
none exists. Use three PLOS folds grouped by topic pair. Source
documents, individual topics and people can still repeat across
folds. Separately lock cutoffs on all PLOS scores and transfer
them to eLife headings. The 20% ceiling is a calibration proxy:
missing tags do not prove a wrong citation. Include the
unfiltered and zero-cutoff operating points. Compare at the
same issuance count as the full reader using test score ranks
without test labels. That last diagnostic assumes access to the
test score distribution and does not define an online policy.
Descriptive uncertainty uses paired bootstrap intervals clustered by
the topic pair.

P95 includes JSON parsing, live embedding, source planning,
three actual warm local declaration reads, any reader forward
and serialization of the proposal and its confidence. Thresholds
are replayed on the measured proposals. There is no network,
serving queue, concurrency, cold start or input indexing cost.

## Results

Both tasks in Slurm array `44486` ran on RTX 5090 on ws5 and
reproduced all non-timing aggregates.
The table reports known supported outputs / topic-unresolved
outputs after the frozen cutoffs or at matched issuance.
All routes inspect the same three source declarations per request.

| Panel | Route | Locked cutoff known / unresolved | At full-reader issuance known / unresolved | Topic checks |
| --- | --- | ---: | ---: | ---: |
| PLOS, 160 | Rank | 29 / 8 | 29 / 3 | 0 |
| | Full | 23 / 9 | 25 / 7 | 960 |
| | Viable | 23 / 11 | 25 / 7 | 175 |
| | Top pair | 33 / 10 | 26 / 6 | 156 |
| eLife headings, 51 | Rank | 12 / 0 | 12 / 0 | 0 |
| | Full | 12 / 0 | 12 / 0 | 306 |
| | Viable | 12 / 0 | 12 / 0 | 46 |
| | Top pair | 12 / 0 | 12 / 0 | 32 |

Relaxing the zero cutoff raises the full-reader PLOS known count
from five to 23 under the locked calibration. On eLife it raises
the full-reader count from zero to 12. On PLOS, the observed
unresolved rate after calibration exceeds the 20% calibration
target in every arm. A development gate is not a guarantee on
held-out data. The rank control produces four more known
results at the same 32 PLOS outputs as the full-reader
confidence ordering. The descriptive interval for the paired
difference per request is −6.54 to +14.12 percentage points.
The equal-issuance diagnostic uses
no topic labels to select which results are returned.
The keyword panel is outside the evaluation.

Restricting the reader to viable pairs cuts PLOS checks from
960 to 175, and model-bearing requests from 160 to 78.
The 156 checks under top-pair verification cannot revise
the preferred structural proof. PLOS local P95 is
9.72–10.18 ms for rank, 41.21–41.26 ms for full,
26.79–27.13 ms for viable, and 22.10–22.16 ms for top
pair. In eLife all four policies give the same 12 known
results under the locked thresholds. On these panels the
rank-based confidence is a stronger efficiency baseline
for the next independent verification experiment.

[Results and paired intervals](results.json)

## Reproduce

Use the two publisher snapshots and checkpoints from the
live local benchmark, on a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.selective_joint_topic_check \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```

Repeat with `--order-offset 1`.
