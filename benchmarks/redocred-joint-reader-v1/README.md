# Verify both conditions on one person in a single prompt

Reuse the pinned calibration and holdout panels and the global
name windows from the
[fixed-budget Re-DocRED experiment](../redocred-conjunctive-proof-v1/README.md).
Each case names two people, two works, and a role. A candidate
qualifies only if source text supports that person's role on both
works. The previously selected two sentences for each atomic
question remain fixed. The experiment changes the reader input
and output format, while source selection and entity resolution
are supplied.

| Route | Rows in one Qwen3-4B batch | Evidence and decision |
| --- | ---: | --- |
| Isolated | 4 | One binary question per relation with its two original sentences. |
| Shared binary | 4 | Both works and the deduplicated sentence table are visible. Each question restricts evidence to the identical two sentence IDs for that work. |
| Joint | 2 | One prompt per candidate with both work-specific claims and their allowed sentence IDs. Four single-token outcomes represent neither, first only, second only, or both. |
| Two-order | 4 | Joint prompt in both work orders. Align the two sets of bit margins and take the smaller margin per relation. |

The shared binary control distinguishes the
effect of exposing the other work's text from the effect of
scoring the outcomes jointly. The model is instructed to use only
the two allowed IDs for each work. That restriction is in the
prompt. No external verifier enforces which text the model uses.

Convert the four joint logits into two atomic log-odds by adding
probability mass for the outcomes that support each claim. The
smaller atomic margin gates a candidate. Also compare the direct
log-odds of “both” against all other outcomes in the same single
joint forward. All methods use the same Qwen3-4B-Instruct-2507
checkpoint and a separate cutoff fitted on the development
windows under the existing 5% unjudged-acceptance ceiling.
Report decisions on the original test and title-disjoint held-out
panels with frozen cutoffs. An unjudged candidate may genuinely
satisfy the request.

Record known conjunctive approvals, unjudged approvals, positive
atomic labels scoring above zero and disagreement under order
reversal. Compare paired differences against isolated inference
by document bootstrap. Report executed padded token slots, which
include instructions and sentence tables. Timing includes selection
of the common windows, prompt assembly, tokenization, inference and
decision computation for a warm local request. It excludes
open-catalog search, networking and service queues.

## Results

Both Slurm tasks in array `44318` completed on an RTX 5090 on
ws5. All non-timing aggregates match. The isolated reader also
reproduces the previous calibration cutoff and approval counts on
all three panels. The table covers 213 held-out cases from 178
documents.

| Reader | Known approved / 160 | Unjudged approved / 266 | Padded token slots | P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Isolated | 50 | 14 | 170,172 | 54.21–54.68 |
| Shared binary | 11 | 31 | 233,520 | 75.83–76.05 |
| Joint, atomic margins | 4 | 3 | 121,020 | 41.46–41.52 |
| Joint, direct both-event margin | 4 | 2 | 121,020 | 41.46–41.52 |
| Two-order agreement | 2 | 5 | 242,040 | 76.44–76.75 |

The joint format saves 28.9% of padded token slots. Its positive
atomic margins before calibration rise from 279 to 364, alongside
a higher threshold on unjudged candidates. After calibration it
loses 46 known conjunction approvals. The paired interval for the
known approval difference per candidate slot is −14.11 to −7.62
percentage points. Holding both possible orders in a single
batch changes the sign of 651 of the 852 atomic comparisons
when the order is reversed. Taking the conservative margin across
orders incurs extra work and leaves two confirmed candidates.
The shared binary control shows a cost and quality penalty from
the shared representation and exposure even before combining the
outputs. These are scoped prompt-format results for a frozen
checkpoint.

### Exact prefix sharing without changing the prompt

A separate CPU audit tokenizes the original isolated prompts.
Of 213 holdout requests, 155 duplicate at least one exact
two-sentence window. A single prefix common to all four prompts
can share 56,907 token positions across the panel. Additional
exact branches can share 18,045 positions, beyond that common
prefix. Under an ideal prefix trie, 158,702 logical token
positions require 83,750 distinct positions. This count covers
exact prefixes only. It excludes cache materialization, branches,
and GPU kernel effects. It identifies a concrete execution
alternative that retains the original prompts, with measured
latency and output agreement still to verify.

[GPU aggregates](results.json) · [Exact prefix audit](prefix.json)

## Reproduce

Use the pinned Re-DocRED dev, test and train files from the linked
benchmark and the same frozen Qwen3-4B model. Run on Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.conjunctive_joint_reader \
  REDOCRED_DIR QWEN4B NEW_RESULTS --order-offset 0
```

Use `PYTHONPATH=src:. python -m experiments.docred.conjunctive_prefix_audit REDOCRED_DIR QWEN4B NEW_AUDIT`
for the CPU token audit.
