# Separate evidence selection from same-person verification

Follow the
[fixed-budget proof window study](../redocred-conjunctive-proof-v1/README.md)
on the same pinned Re-DocRED panels. Use the supplied people,
works and relation. Evaluate every known candidate for whom both
relations have published sentence annotations and each annotated
set fits within two sentences. The native test contributes three
such candidates. The larger title-disjoint holdout contributes
56 candidates. These are diagnostic subsets of positive
conjunctions.

Start from the global-name selector's windows. Each candidate has
two atomic questions concerning the same person and two different
works. Intervene only on the tested candidate's windows:

| Route | Change to the retrieved windows |
| --- | --- |
| Retrieved | Keep both original windows. |
| First proof | Give the first question its full annotated sentence set. |
| Second proof | Give the second question its full annotated sentence set. |
| Both proofs | Give both questions their respective annotated sentence sets. |

Keep two sentences per question. If a gold set contains one
sentence, fill from the original window by the frozen lexical
score. The other person's two question contexts remain untouched.
There are four model comparisons in one batch per route, with
two sentences per comparison. The intervention can change the
total number of distinct sentences across the two people, and
token counts and padding lengths can differ.

Compare frozen Qwen3-4B-Instruct-2507 with frozen
Qwen3-Reranker-0.6B. The margins are the models' support scores.
For each reader, fit a single conjunction threshold on the
original global-name calibration windows, allowing at most
5% of candidate conjunctions without a complete relation label.
A candidate passes only when both atomic scores exceed that
cutoff. Apply each threshold unchanged to all intervention
views. Report positive decisions and paired document intervals.
Track changes in the other, unchanged candidate's decision to
audit cross-prompt execution drift.

On candidates with disjoint annotation sets for the two works,
also give the two questions the same annotated set. Exclude the
other work's annotations from filler. Count how often the
eligible candidate still passes after the swap, conditional on
passing when shown both annotated sets. Other released text
could imply the missing relation. Treat the swap as a reader
sensitivity probe rather than a confirmed negative label.

Annotation insertion is a diagnostic oracle. An improvement
isolates the effect of making the annotated text available
under the same per-claim budget. A residual rejection can
reflect insufficient standalone context, score calibration, or
a reader error. It is not an estimate of the real-world
positive rate or an upper bound on what a different model or
window could recover. Timings cover the evidence formatting and
reader batch, without corpus search or service overhead.

## Results

Both Slurm tasks in array `44313` completed on an RTX 5090 on
ws5 and reproduced all non-timing counts. The 4B threshold
matches the previous global-name experiment. Both readers accept
two of 52 unjudged conjunctions on the calibration windows. The
table covers the 56 annotated conjunctions in the larger
holdout.

| Reader | Retrieved | Insert first | Insert second | Insert both | Both-proof P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 14 | 21 | 17 | 25 | 53.92–53.96 ms |
| Qwen3-Reranker-0.6B | 9 | 12 | 9 | 12 | 12.71–12.77 ms |

On the original windows, 30 candidates already have both
annotations, 23 have one, and three have neither. The 4B reader
accepts 11 of the 30 with both annotations already present. Its
11 added acceptances after full insertion consist of ten
recoveries in the one-missing group and one in the two-missing
group, with no losses. The paired document-bootstrap interval
for the 4B improvement is +7.14 to +32.73 percentage points.
The 0.6B reader gains four acceptances and loses one. Its paired
change interval is −2.08 to +15.25 points. The other candidate's
binary decision remains unchanged in every diagnostic view.
The native test has three candidates; 4B approves two before and
after insertion, while the 0.6B count rises from zero to one.

All 56 conjunctions receive both annotation sets in the last
column. At the frozen thresholds, 31 still fail the 4B decision
and 44 fail the 0.6B decision. This separates a recoverable
selection loss from the remaining gap under short-window
verification. The remaining gap includes any dependence on
context outside the released annotation, as well as reader and
threshold errors. The result does not provide a calibrated
false-positive rate for an intervention on ordinary requests.

Seven candidates have disjoint annotation sets. Of the three
that the 4B model accepts with both sets, all three fail when
either work's annotated set is replaced by the other's. The
0.6B model accepts four with both sets. Three fail after the
first set replaces the second, and all four fail in the reverse
substitution. These small controls are evidence that the
reader can respond to changes in work scope. They are weak
evidence about general cross-work confusion.

[Both GPU aggregates](results.json)

## Reproduce

Use the pinned snapshot and checkpoints in the
[preceding benchmark](../redocred-conjunctive-proof-v1/README.md).
Run in an allocated Slurm CUDA job.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.conjunctive_proof_injection \
  REDOCRED_DIR QWEN4B QWEN_RERANKER NEW_RESULTS --order-offset 0
```
