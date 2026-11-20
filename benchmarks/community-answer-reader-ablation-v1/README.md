# Diagnose verifier loss after three-source community retrieval

Freeze the three answer selections of the
[community allocation benchmark](../community-joint-answer-witness-v1).
Do not let rescoring change the staged policy's third answer. Each run
must reproduce all original panel aggregates and original zero-cutoff
outputs before the ablation is accepted.

## Diagnostics

1. **Answer window.** Compare the 1,800-character answer prefix with
   passing the entire answer to the same verifier. Both variants use
   the same 1,024-token packed context limit, so long inputs can still
   be clipped. Compare the same ordered question–answer pairs.
2. **Decision threshold.** Sweep the prespecified yes-minus-no logit
   margins `-4, -2, 0, 2, 4`. The weaker answer margin must exceed
   the cutoff. Hold expert attribution, proposal eligibility and
   ranking fixed within each window setting. For a selected answer
   set with a known proof, distinguish cases where no known proof
   passes from cases where a passing known proof ranks below another
   proof. Alternative cutoffs are diagnostics on the same panels.
3. **Question wording.** On known Cross Validated question–answer
   pairs, replace the later duplicate question with the original
   canonical question on which that answer was accepted. Keep the
   answer prefix and threshold unchanged. This labeled-positive
   comparison is not a deployable action, and the two questions may
   differ in specificity, length and information content. MathOverflow
   already uses original parent questions.

Neither the accepted-answer judgments nor this threshold sweep estimate
precision for general requests. An issued answer pair outside the known
labels remains unresolved. Overlapping questions and candidates in the
composed cases make counts descriptive.

## Results on RTX 5090

Both tasks of Slurm array `44698` on `ws5` reproduced all report counts.
In each cell, the two numbers are known returns / unresolved returns.
No-issuance requests are outside those two counts. All comparisons
reuse the same answer sets.

| Panel and policy | Prefix, cutoff 0 | No character cutoff, cutoff 0 | Prefix, cutoff −2 | Prefix, cutoff −4 |
| --- | ---: | ---: | ---: | ---: |
| Cross Validated 2024, staged | 24 / 0 | 26 / 0 | 40 / 1 | 48 / 16 |
| Cross Validated 2024, joint | 24 / 0 | 27 / 0 | 43 / 0 | 52 / 22 |
| Cross Validated 2025, staged | 6 / 0 | 6 / 0 | 6 / 0 | 9 / 2 |
| Cross Validated 2025, joint | 5 / 0 | 5 / 0 | 7 / 0 | 14 / 9 |
| MathOverflow, staged | 66 / 0 | 69 / 0 | 83 / 5 | 89 / 5 |
| MathOverflow, joint | 65 / 0 | 68 / 0 | 85 / 4 | 90 / 4 |

At the original cutoff, joint selection exposes known proofs in
80, 22 and 109 requests. In 56, 17 and 44 of those requests,
respectively, no known proof passes both conditions. There are
zero cases in any policy where a known proof passes but loses the
ranking at that cutoff. Removing the character cutoff adds 0–3
known returns per panel and policy. Model-context clipping still
occurs, so this comparison covers the application cutoff only.

The less conservative cutoff of −2 increases known recovery with
few unresolved outputs in this positive panel. At −4, unresolved
joint results increase to 22 and 9 in the Cross Validated panels.
The exploratory contrast estimates the sensitivity of the
verification rule. Choosing a deployment cutoff requires a separate
development split and a held-out panel that includes independently
judged nonmatches and incomplete evidence.

For the same known answer, the original canonical question passes
where the later duplicate fails in 10/29 and 8/16 unique
question–answer pairs. The opposite happens in 2/29 and 1/16.
This is consistent with sensitivity to differences between questions.
The acceptance record alone cannot distinguish a verifier error
from a difference in what the questions require.

[Aggregate results](results.json).

## Reproduce

Use the frozen snapshots and checkpoints from the allocation benchmark.
Run on a Slurm GPU with a fresh output path. Repeat with
`--order-offset 1`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.community.answer_reader_ablation \
  STATS_2024 STATS_2025 MATHOVERFLOW QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/community-joint-answer-witness-v1/results.json NEW_RESULTS
```
