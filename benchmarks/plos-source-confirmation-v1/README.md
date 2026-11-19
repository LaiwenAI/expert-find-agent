# Text confirmation after identity-linked source expansion

Use all 25 exact-ORCID paper pairs in the 171-paper [source-discovery corpus](../plos-scoped-search-v1/README.md). The pairs connect 36 papers grouped into 15 identity-linked components. The task provides two source clues per request. Use result excerpts and keyword views, each with both sources present or one specified source masked. This yields 50 present cases and 100 missing-source cases. Some underlying papers and clues have been used in previous studies.

Compare a top-four candidate join with expanded search through rank 12, where one source must still be within rank four. Use the same candidate generators in all routes. A separate text gate assesses only the best expanded pair. It uses the worse of two dense ranks, the lower Qwen3-Reranker source logit, or the lower value from source logits and gaps against stronger individual fused-rank alternatives. The cross-encoder receives only the source clue and indexed text. Author identities and contributor-role labels do not enter its prompts.

Choose a strict acceptance threshold above the largest false-candidate score on the remaining components, for each held-out component. Count false candidates when the required source is absent and when an incorrect pair is found with sources present. The selective variants preserve every top-four answer and calibrate only on proposals where top-four abstains. Evaluate held-out folds together for a descriptive comparison. With 15 components, the results characterize this corpus and protocol. A zero observed false-acceptance rate on calibration does not imply a risk guarantee.

## Results

| Source policy | Exact among 50 present | Wrong accepts among 50 present | False accepts among 100 missing | Cross-encoder calls / 150 |
| --- | ---: | ---: | ---: | ---: |
| Linked top-four | 30 | 1 | 3 | 0 |
| Linked expansion | 40 | 1 | 8 | 0 |
| Dense rank gate on all expanded proposals | 1 | 0 | 1 | 0 |
| Absolute text gate on all expanded proposals | 20 | 0 | 1 | 49 |
| Relative text gate on all expanded proposals | 19 | 0 | 1 | 49 |
| Selective dense rank gate | 30 | 1 | 4 | 0 |
| Selective absolute text gate | 31 | 1 | 4 | 15 |
| Selective relative text gate | 30 | 1 | 4 | 15 |

Expanded retrieval finds ten more correct pairs than the narrower join and accepts five more missing-source substitutes. Applying the absolute text gate to every expanded proposal removes seven missing-source accepts while discarding twenty correct pairs. Preserving top-four decisions and gating only expansions reduces the number checked from 49 to 15. It recovers one additional true pair and accepts one additional missing-source substitute relative to top-four. The selective policies are exploratory follow-ups on the same corpus. On the keyword subset, the selective text gate recovers none of the five additional true pairs returned by expansion.

On one RTX 5090, common retrieval P95 is 9.05 ms. The separately timed confirmation batch P95 is 13.91 ms among checked pairs. Factoring that timing into queries where the selective absolute route calls the model gives a 21.19 ms P95 estimate, compared with 9.05 ms for common retrieval alone. This estimate does not come from independently timed live executions of all policies, and retrieval timing includes computing both graph joins. It excludes role parsing and expert ranking.

The tested confirmation scores show a recall–risk tradeoff. They have not established an improvement over the narrower join at equal observed risk. Report source identity, missing-source abstention and the cost of confirmation before evaluating the full recommendation. [Aggregate results](results.json)

## Reproduce

Use the [published fixture](../plos-credit-source-scoped-v1/fixture.json), original XML inputs, and the [pinned source projection](../plos-scoped-search-v1/README.md). Rebuild the panel with the capture directories used by the source-discovery experiment. The run command uses an open-weight embedding model and reranker on a Slurm CUDA node.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m experiments.credit.scoped_source_confirmation prepare \
  SCOPED_FIXTURE SOURCE_PROJECTION NEW_PANEL XML_DIRECTORY [XML_DIRECTORY ...]
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.scoped_source_confirmation run \
  SCOPED_FIXTURE SOURCE_PROJECTION SCOPED_PANEL QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
