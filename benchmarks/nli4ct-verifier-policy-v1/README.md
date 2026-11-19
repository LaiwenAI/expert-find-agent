# Verifier-aware evidence depth with a single reader call

The [coverage-trained controller](../nli4ct-adaptive-depth-v1/README.md) recovered more complete evidence at a lower sentence count than fixed two-step expansion. The [live reader study](../nli4ct-adaptive-reader-v1/README.md) found lower binary accuracy despite that coverage gain. This experiment keeps the same candidate expansion order and learns which depth to send to the reader.

For each query, run the pinned Qwen3-4B reader on the initial context and on up to three successive expansions. Reuse the original trial-disjoint splits with 1,200 train, 200 development and 500 test statements. A ridge model predicts the change in binary reader correctness at each depth relative to two expansions, using only text and controller features at inference time. Candidate features include query terms covered in each source, extent of the proposed view, prior expansion scores and a word-count cost. Choose one depth before the reader runs, using predicted correctness gain minus a price on additional words.

Set the price on the development split. Minimize mean measured reader input tokens subject to an observed accuracy drop of at most one percentage point relative to two expansions. Include fixed two-step expansion as a fallback in the calibration grid. Freeze the selected price and coefficients before the test. The two-expansion arm stays available to the policy for each statement. The one-percentage-point development tolerance is an operating-point choice and is not a test noninferiority bound. Supplied trial IDs and section names are shared across arms.

## Results

| Route | Correct / 500 | Mean input tokens | Live P95 |
| --- | ---: | ---: | ---: |
| Initial windows | 301 | 290.96 | 35.27 ms |
| Fixed two expansions | 322 | 376.95 | 41.15 ms |
| Fixed three expansions | 324 | 410.10 | 44.82 ms |
| Coverage-trained stopping | 311 | 348.02 | 37.70 ms |
| Verifier-aware depth | 302 | 291.50 | 35.50 ms |

Calibration selects the largest tested price. The selected policy and two expansions both classify 132 of 200 development statements correctly. The policy sends 461 of 500 test statements to the initial window. Compared with two expansions it reduces mean prompt tokens by 22.7% and observed P95 by 13.7%. Test accuracy falls by 4.0 percentage points, with a 95% paired interval of −6.9 to −1.4 points across connected trial components. The lowest-cost policy is close to the initial-window baseline. Simple pre-reader features and this development restriction did not reliably identify which requests could skip reading.

The size of the initial-window penalty shifts across splits. Initial windows classify 133/200 correctly on development and 301/500 on test. Two expansions classify 132/200 and 322/500 respectively. A post-hoc oracle that always picks the shortest view preserving every fixed-two correct judgment uses 296.91 input tokens on average. This is an upper bound on the available token savings under that requirement, derived from known test judgments. It indicates room for query-level routing without establishing that a deployable score can achieve it.

Job 41386 generated trajectories and job 41391 ran the live comparison on an RTX 5090 on ws5. The live path includes evidence selection, prompt construction, tokenization and GPU inference at batch size one from supplied sections. Independent CPU training reproduced the selected operating point and replay outcomes. [Training and replay](policy.json) · [Live aggregates](live.json)

## Reproduce

Use the [pinned NLI4CT archive](https://github.com/ai-systems/nli4ct/blob/8e8e5fc506b916984011873403b9ea5ed6dd96c2/Complete_dataset.zip), the [frozen controller](../nli4ct-adaptive-depth-v1/results.json) and the [pinned Qwen checkpoint](../scifact-live-selective-v1/README.md). Run trajectory generation and live verification within Slurm CUDA allocations. The training command runs on CPU. Use a new path for every output.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.scifact.nli4ct_verifier_traces \
  NLI4CT_ARCHIVE FROZEN_CONTROLLER QWEN_4B_CHECKPOINT NEW_TRACES
python -m experiments.scifact.nli4ct_verifier_policy train \
  NEW_TRACES FROZEN_CONTROLLER NEW_POLICY
python -m experiments.scifact.nli4ct_verifier_policy live \
  NLI4CT_ARCHIVE FROZEN_CONTROLLER NEW_POLICY QWEN_4B_CHECKPOINT NEW_LIVE_RESULTS
```
