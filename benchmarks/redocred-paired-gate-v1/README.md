# Contrast-trained substitution at a fixed verification budget

Test whether a learned gate preserves the benefit of grammatical
role links while limiting the number of additional acceptances
that need human review. Use a linear decision model over raw
candidate and query features. The model compares the last
shortlisted person with an alternative that has a stronger
grammatical role link. It keeps the original slot unless the
score margin clears a threshold. Each arm uses the same
open-weight Qwen reader, pair count, and single model forward.

## Construct the contrast data

Construct rows from the public Re-DocRED development split.
An original row has a labeled person–work–role relation. The
counterpart has a labeled competing role and lacks a target
relation judgment. It serves as a benchmark rival. Unjudged
candidates with no competing-role label are excluded from fitting.

- **Person swap:** Hold the work, source, and requested role fixed.
  Pair a judged positive with a different person credited for
  another role in that source.
- **Role swap:** Hold the work, source, and resolved person fixed.
  Pair a supported role with a different requested role for which
  the person is a labeled rival.

The pointwise control and the contrast learner use the same
judged rows, features, architecture, and optimizer. The pointwise
control minimizes binary cross-entropy. The contrast learner adds
two within-pair ranking losses, one per swap type. Both are trained
on the allocated GPU. The reader weights are frozen. The contrast
manifest records candidate strings and document indices into the
public dataset, together with the contrast types.

## Calibrate and compare

Fit only with the upstream dev split. Choose each learner's margin
on the upstream test split, with works already used for fitting
excluded. Compare each threshold with keeping the original
shortlist and applying the syntax-based window rule. Maximize
the number of additional covered positives subject to no increase
in accepted known rivals and no more than five additional
unjudged acceptances per 100 calibration queries. This allowance
is a review workload constraint. It supplies no assertion about
the correctness of unjudged results.

Use the same 754 exact queries and 623 token-drop variants as the
[grammar-link experiment](../redocred-syntax-v1). Their reference
works are outside the fitting and calibration panels. This panel
was previously used to compare grammatical role links and serves
as a follow-up comparison. Report the two views separately.
Their reference documents overlap.

| Method | Last candidate | Evidence windows |
| --- | --- | --- |
| Single batch | Original | Lexical |
| Syntax window | Original | Include an attached role sentence when needed |
| Syntax joint | Grammatical rule | Syntax |
| Pointwise | Calibrated pointwise model | Syntax |
| Contrasts | Calibrated contrast-trained model | Syntax |

A covered positive requires acceptance of a labeled correct
person–work–role relation and inclusion of all its annotated
evidence sentences. Report known competing-role acceptances and
unjudged acceptances separately. Paired intervals cluster by
reference document. Warmed serial P95 includes routing, online
candidate decisions, window selection, and the real reader
forward. It excludes model loading and offline parsing.

## Results

Slurm job 43207 completed on one RTX 5090 on ws5. The shared
training set contains 744 candidate records, 773 person swaps
and 653 role swaps. Each learned model has 26 parameters.
On 297 calibration queries, the pointwise gate adds two covered
positives and eight unjudged acceptances. The contrast-trained
gate adds one covered positive and one unjudged acceptance.

| View | Method | Covered positives | Accepted positives | Known rivals | Unjudged | Replacements | P95, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | Single batch | 296 | 464 | 6 | 62 | 0 | 61.09 |
| Exact | Syntax window | 298 | 466 | 6 | 62 | 0 | 61.19 |
| Exact | Syntax joint | 311 | 479 | 5 | 136 | 276 | 61.37 |
| Exact | Pointwise | 306 | 474 | 6 | 73 | 102 | 61.54 |
| Exact | Contrasts | 298 | 466 | 6 | 62 | 2 | 61.54 |
| Token drop | Single batch | 110 | 193 | 2 | 99 | 0 | 61.64 |
| Token drop | Syntax window | 111 | 195 | 2 | 99 | 0 | 62.09 |
| Token drop | Syntax joint | 115 | 198 | 2 | 128 | 312 | 62.14 |
| Token drop | Pointwise | 112 | 197 | 2 | 113 | 178 | 62.21 |
| Token drop | Contrasts | 111 | 196 | 2 | 99 | 4 | 62.06 |

For exact names, the pointwise gate preserves eight of the
13 extra covered positives that the syntax rule adds over
keeping the original shortlist with syntax windows. It leaves
11 extra unjudged acceptances, down from 74 with the syntax rule.
Against the syntax rule, its covered P@5 difference is −0.13
percentage points (document-clustered 95% interval −0.33 to
+0.05). The observed recall and review-volume trade-off does
not establish noninferiority at a prespecified margin. Under
token drop it preserves one of the four extra covered positives
and leaves 14 extra unjudged acceptances, down from 29.
The pointwise run preserves the single forward and adds less
than 0.6 ms to serial P95 relative to the original single batch
in both panels.

The calibrated contrast gate changes two exact-name slots and
four token-drop slots. It yields the same covered-positive
counts as the syntax-window-only control. The extra contrast
loss did not improve this candidate substitution policy.
A useful next comparison would model whether a specific
replacement adds verified coverage relative to keeping the
incumbent, with additional independent annotations for
candidates whose relationships are currently unjudged.

[Aggregate results](results.json) · [Public training contrast manifest](training-contrasts.json)

## Reproduce

Use the same pinned Re-DocRED data, spaCy runtime and parser,
and Qwen reader listed in the [grammar-link experiment](../redocred-syntax-v1).
Run on Slurm CUDA with a fresh output directory.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.paired_role_gate \
  path/to/pinned/snapshot path/to/train_revised.json \
  path/to/qwen3-4b-instruct-2507 path/to/new-results.json \
  c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2
```
