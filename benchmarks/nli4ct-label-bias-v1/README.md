# Test whether output-label bias limits evidence verification

Run the same 500 NLI4CT test statements, frozen Qwen3-4B reader, and fixed-depth and annotated-only input views from the [evidence intervention](../nli4ct-evidence-intervention-v1/README.md). Hold the evidence, model, and section formatting fixed. Independently swap the digits assigned to entailment and contradiction, and reverse the order in which their meanings are listed. Keep digit 2 assigned to insufficient evidence in all prompts. Compare forced binary verdicts after mapping the logits back to their semantic labels. The original condition reproduces the earlier prompt exactly. [Related work on choice-symbol bias](https://aclanthology.org/2025.naacl-long.95/)

Average the original and digit-swapped margins as a two-call control. Also compare averaging all four factorial variants. For a one-call alternative, estimate the mean half-gap between the digit-swapped and original entailment-minus-contradiction margins on the separate 200-statement development split. Then add this fixed offset to the original prompt's test margin. This takes no development target labels. Estimate a separate offset per evidence view. The runner checks that calibration and test share no report IDs. Frozen offsets incur no additional model forward at inference time.

The runner measures single-call prompt construction, tokenization and inference. Extra-call averages are quality controls, and their serving latency is not inferred from component P95s. Test confidence intervals resample connected trial components. This compares model behavior on public benchmark statements from supplied sections and does not validate clinical decisions or expert matching.

## RTX 5090 results

The test set has 250 entailment and 250 contradiction statements.

| Reader policy | Fixed two correct / 500 | Annotated only correct / 500 | Inference forwards |
| --- | ---: | ---: | ---: |
| Original ordering and digits | 322 | 330 | 1 |
| Swap digits | 340 | 350 | 1 |
| Reverse display order | 319 | 319 | 1 |
| Swap digits and display order | 354 | 371 | 1 |
| Fixed offset on the original margin | 322 | 330 | 1 |
| Average original and swapped-digit margins | 338 | 355 | 2 |
| Average all four margins | 363 | 370 | 4 |

Job 42014 reproduces the original reader controls. Reversing display order alone changes 271 fixed-depth and 259 annotated-only decisions, and increases predicted entailments from 114 to 385 and from 134 to 393, respectively. Swapping both digits and order improves fixed-depth accuracy by 6.4 percentage points (paired, connected-component 95% interval +2.50 to +10.09) and annotated-only accuracy by 8.2 points (+3.54 to +12.02). Single-call P95 for the fixed-depth view is 40.41 ms for the original prompt and 40.30 ms for the combined swap. This prompt variant is a candidate identified within this test and needs an independent replication before use as a default.

A constant offset estimated from the development set changes some verdicts but gives no net accuracy gain. The four-variant mean improves fixed-depth accuracy by 8.2 points (+4.27 to +11.74), with four calls. Its binary predictions still miss over a quarter of the cases. Prompt changes isolate sensitivity to label presentation under fixed evidence, but this diagnostic does not assign all remaining errors to a specific cause. Intervals are descriptive and are not adjusted for multiple comparisons. [Aggregate results](results.json)

Run on a Slurm CUDA node with the same pinned public archive, frozen controller, and local open-weight checkpoint as the [evidence intervention](../nli4ct-evidence-intervention-v1/README.md).

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.nli4ct_label_bias \
    NLI4CT_ARCHIVE FROZEN_CONTROLLER READER_CHECKPOINT NEW_RESULTS
```
