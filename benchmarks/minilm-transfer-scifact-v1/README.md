# Transfer trained evidence-verifier weights from VitaminC to SciFact

Load the frozen multilingual MiniLM and six verified checkpoint subsets from the [VitaminC objective comparison](../vitaminc-risk-tail-minilm-v1/README.md). The runner checks every tensor file hash against the training aggregate. No gradient updates run on SciFact data.

Reuse the [SciFact evidence-window study](../scifact-nli-windows-v1/README.md) sampling rule. It gives 309 calibration and 226 evaluation claim–abstract pairs with disjoint publication IDs and one pair per claim and publication within each split. This benchmark has appeared in earlier research. For each pair compare the full supplied abstract and a three-sentence lexical window selected without gold annotations. A directional label comes from released SciFact evidence. For the unknown class, use the task's cited abstracts without released evidence. Those abstracts are a proxy for insufficient evidence. Window metrics retain the full-pair label even when the window misses an annotated rationale, so additionally count support acceptance on windows containing a complete rationale.

Apply each model's original VitaminC cutoff from the strict development setting and a second cutoff calibrated on the separate SciFact calibration panel. Both calibrations constrain false support on refutation and on insufficient evidence separately. The target-domain threshold uses up to 5% of the negative examples in each class; this criterion does not guarantee the same error count at test. For each cutoff report the test support and both types of false support. Timing starts from the supplied view and includes tokenizer and model batch size one. It excludes retrieval and window selection.

## RTX 5090 results

Test counts are true supports / false accepts on refutations / false accepts on unknowns, with denominators 97 / 47 / 82. All full-abstract arms abstain on the same 29 over-limit inputs. The lexical windows fit for all 226 pairs.

| Training arm | Seed | Full, source cutoff | Full, SciFact cutoff | Three sentences, source cutoff | Three sentences, SciFact cutoff |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen | — | 2 / 0 / 0 | 8 / 2 / 5 | 3 / 0 / 0 | 31 / 3 / 3 |
| Classification | 20261119 | 8 / 4 / 1 | 8 / 4 / 1 | 9 / 0 / 0 | 20 / 0 / 2 |
| Within-pair | 20261119 | 8 / 4 / 1 | 8 / 4 / 1 | 9 / 0 / 0 | 21 / 0 / 2 |
| Upper-tail | 20261119 | 10 / 4 / 1 | 15 / 5 / 3 | 9 / 0 / 0 | 26 / 1 / 3 |
| Classification | 20261120 | 8 / 4 / 1 | 8 / 4 / 1 | 9 / 0 / 0 | 20 / 0 / 2 |
| Within-pair | 20261120 | 8 / 4 / 1 | 9 / 4 / 1 | 9 / 0 / 0 | 22 / 0 / 2 |
| Upper-tail | 20261120 | 6 / 3 / 2 | 15 / 4 / 3 | 8 / 0 / 0 | 28 / 1 / 4 |

With the target-domain threshold, upper-tail training recovered seven additional supported claims on full abstracts in both seeds and increased test refutation errors. The paired interval for the full-abstract increase was +3.09 to +12.37 percentage points in each seed. With three sentences it reduced refutation errors from three to one and retained three to five fewer true supports than frozen MiniLM. Among the 58 lexical windows containing a complete annotated support rationale, frozen accepts 23 and the two upper-tail seeds accept 22 and 23. The classification and within-pair controls accept 16 to 18. Moving the VitaminC cutoff directly gives much lower lexical-window support coverage across all models.

The job uses a single reader call with lexical-window P95 of 1.31–1.37 ms across arms. It demonstrates sensitivity to the evidence view and threshold transfer. These thresholds and partial labels do not establish a risk-constrained quality gain or expert recommendation accuracy. [Aggregate results](results.json)

Generate the verified saved weights from the linked VitaminC study and run the evaluation with the pinned [SciFact inputs](../scifact-nli-windows-v1/README.md) inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.minilm_transfer \
    SCIFACT_ARCHIVE VITAMINC_RESULT VITAMINC_WEIGHTS MINILM_CHECKPOINT NEW_RESULT
```
