# Resolve two roles in one verifier forward pass

The [factorial diagnosis](../source-role-factorial-diagnosis-v1)
found that extra plausible roles frequently blocked otherwise
correct results. Hold the matched source pair and source-search
branch fixed. If a source check has used the one-forward budget,
do not call the role reader. Preserve a proof that already passes
the ridge-band check. Otherwise score three candidate labels per
activity by the mean standardized ridge and prototype scores.
Skip the role read if no contributor could satisfy any candidate
label on both papers.

Compare the deployed selective single-clause reader with these
four two-slot methods:

- **Dense top one** chooses each shortlist leader without a role
  model forward.
- **Two serial** classifies each phrase with the original single-slot
  prompt in a separate forward pass. It is an over-budget control.
- **Two batched** applies the same two prompts and alternatives as
  two rows of one forward pass.
- **Packed pair** puts both activities, the shared role definitions,
  all nine ordered combinations, and a no-match option in one
  prompt and reads a one-token choice.

Prompts contain the activity text and role definitions. They omit
paper, author and applicability data. A selected pair narrows each
role slot to one label. The same-owner proof still requires both
roles to be explicitly declared on the respective sources.
Calibrate the minimum logit gap for independent predictions or the
pair logit gap for packed predictions on the frozen 48 PLOS cases.
Select the threshold maximizing strict successes under the original
5% case-level adverse allowance, breaking ties by fewer adverse
cases and then by higher cutoff. Target cases select no settings.

Replay the same 27 positive requests and 27 role swaps from the
last diagnosis with supplied keyword and excerpt spans. Include
left- and right-source removal controls. Check exact baseline
response parity with the live service handler when both sources
are available. Report forward dispatches and nonpadding role-prompt
tokens separately. Local P95 includes resident text embedding,
source selection and any model call. It excludes HTTP, queueing,
index build and natural-language span extraction.

## Results

All tasks in arrays 46134 and 46140 completed on ws3 RTX 5090.
Final counts reproduce in reverse request order. Each view has
54 requests, including 27 positives; the same base queries recur
across views. The role-forward count excludes source checks.

| Route | Keyword strict / adverse / issued | Excerpt strict / adverse / issued | Role forwards, keyword / excerpt | Role tokens, keyword / excerpt |
| --- | ---: | ---: | ---: | ---: |
| Single-slot baseline | 15 / 0 / 15 | 17 / 0 / 17 | 10 / 12 | 842 / 1030 |
| Dense top one | 15 / 3 / 18 | 17 / 3 / 20 | 0 / 0 | 0 / 0 |
| Two serial | 21 / 1 / 22 | 23 / 1 / 24 | 56 / 72 | 4400 / 5652 |
| Two batched | 21 / 1 / 22 | 23 / 1 / 24 | 28 / 36 | 4400 / 5652 |
| Packed pair | 8 / 0 / 8 | 10 / 0 / 10 | 28 / 36 | 5627 / 7266 |
| Reserve batch | 20 / 1 / 21 | 22 / 1 / 23 | 28 / 37 | 3656 / 4937 |
| Reserve batch with dense agreement | 17 / 0 / 17 | 19 / 0 / 19 | 28 / 37 | 3656 / 4937 |

The reserved variants are follow-up diagnostics on the same panel.
They use the original single-slot route when a pre-read branch can
complete a proof with one clause, and use the batched rescue only when
that route would not call. The agreement variant additionally requires
both batched labels to match the dense top choices. Gates still come
from the original development data; the extra policies were introduced
after inspecting the first transfer experiment.

Batched predictions give the same aggregate results as serial
predictions and halve role forward dispatches for the same number of
nonpadding role tokens. Across panels and views, serial P95 is
31.4–36.9 ms, with batching at 25.1–30.1 ms. Relative to the single-slot
baseline, the batched option increases role calls and prompt tokens.
Its extra coverage comes with one adverse issuance per view.
The dense agreement variant retains two additional strict proofs per
view with zero observed adverse issuance. The held-source controls
observe no wrong-source issuance across the 216 cases per route.
The panel and amount of issued evidence remain small. A prospective
comparison needs new paper pairs and real service workloads to
estimate quality, latency and cost jointly. [Complete results](results.json)

## Reproduce

Use the inputs in the [factorial diagnosis](../source-role-factorial-diagnosis-v1)
and add the PLOS projection immediately after the engineering
manifest. Change the module to
`experiments.credit.paired_role_verification`. Run both request
orders in Slurm CUDA, using `--reverse` and a fresh output path
for the repeat.
