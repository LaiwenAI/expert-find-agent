# Learn to assemble proof pairs under a fixed reading budget

Reuse the pinned Re-DocRED panels, two people by two works, and
the Qwen3-4B reader from the [proof-window evaluation](../redocred-conjunctive-proof-v1).
Two relations about the same person must hold. Every route passes
four distinct original sentences across exactly four two-sentence
prompts. The experiment holds document discovery and entity identity
fixed. The number of tokens may differ.

Train a linear pair scorer on the remaining upstream training titles
after excluding every title in the held-out panels and in the official
development and test releases. A comparison prefers a two-sentence
window that contains all annotated evidence for a relation over a
lexically strong pair that omits some of the annotation. Incomplete
windows are ranking alternatives, and may still convey valid evidence.
Pair features capture exact and partial entity mentions, role cues,
local co-occurrence, competing names, distance and position. Fit ridge
least squares to feature differences with equal total weight per
eligible claim. No reader tuning or synthetic labels are used.

| Route | Selection rule |
| --- | --- |
| Global names | Original greedy name coverage followed by lexical assignment |
| Lexical pair | Maximize the sum of lexical pair scores |
| Learned sum | Maximize the sum of learned pair scores |
| Learned conjunction | Maximize the sum of each candidate's weaker claim score, breaking ties by the learned sum |

The pair selectors propose shared windows from unions of two
shortlisted pairs, fill remaining slots from the original global
window, and retain the original window as a proposal. Evaluate every
assignment of pairs within each proposed window that uses all four
sentences. Learned sum and learned conjunction use the same proposal
set, so their comparison isolates the allocation objective. Learned
sum and lexical pair use the same sum objective and proposal procedure,
with a different scoring function.

Report complete annotated evidence for each person, and fixed issuance
of one candidate based on the smaller of the two reader margins
(ties use the first candidate in the stable source order). Separately
report thresholded approvals to show the effect of abstention. Fit
one threshold per arm with the same development ceiling of 5%
unjudged approvals. Missing labels remain unjudged. Sentence coverage
is an annotation-overlap measure and the label for a documentary role
does not establish expert suitability. Compare policies with intervals
clustered by source title. Latency covers window selection and live
reader inference, outside queues and open-catalog discovery.

The method allocates a single evidence budget across competing
conjunctions. [Eider](https://aclanthology.org/2022.findings-acl.23/)
learns evidence extraction for document-level relations and combines
local and full-document predictions. The comparison here isolates
allocation and holds the verifier and number of submitted sentences
fixed.

## Results

Both tasks of Slurm array `44359` ran on an RTX 5090 on ws5 with
rotated arm order. Every non-timing statistic reproduced. The ranking
weights fit 25 annotated claims from nine disjoint source titles.
The larger held-out panel has 213 requests from 178 titles and
73 candidate conjunctions with both sentence annotations.

| Selector | Complete annotated proof / 73 | Known top one / 213 | Top one with complete proof / 213 | Thresholded known / unjudged | Selection and verifier P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global names | 30 | 93 | 25 | 50 / 14 | 53.64–53.95 |
| Lexical pair | 21 | 99 | 18 | 56 / 16 | 53.62–54.20 |
| Learned sum | 38 | 98 | 32 | 51 / 11 | 53.97–54.27 |
| Learned conjunction | 40 | 102 | 32 | 33 / 7 | 53.85–54.25 |

At fixed issuance the learned conjunction adds nine known top choices
against global names (+4.23 percentage points, paired 95% percentile
interval +0.47 to +8.14). The contrast against the lexical pair arm is
three more known top choices and 14 additional top choices with both
evidence annotations. The learned sum ties the conjunction objective
on top choices with both annotations. On the 38 official test
requests, global names and both learned routes all retrieve 16 known
top choices. The cohorts contain previously studied documents, and the
intervals describe exploratory paired contrasts.

Thresholded decisions matter for a system that can abstain. At the
unchanged development calibration procedure, the learned conjunction
accepts only 33 known candidates versus 50 for global names, and
accepts seven unjudged candidates versus 14. The held-out operating
points therefore have different issuance. Conjunction selection merits
a larger-data follow-up with frozen common-coverage comparisons before
any claim of better calibrated recommendations. Selection alone takes
about 1 ms P95 for either learned route. Combined verification latency
is similar in this local four-prompt workload.

[Complete aggregate results for both runs](results.json)

## Reproduce

Use the same data and checkpoint as the proof-window evaluation.
Run in a Slurm GPU allocation, with a fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.learned_conjunctive_windows \
  REDOCRED_DIR NEW_RESULTS --checkpoint QWEN4B --order-offset 0
```

Repeat with order offset 1. Omitting the checkpoint runs the
annotation-overlap comparison without model inference.
