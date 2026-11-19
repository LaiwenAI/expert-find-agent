# Train a compact verifier on short evidence windows

Use the pinned SciFact archive and paper-disjoint split from the [earlier supervised verifier experiment](../scifact-nli-finetune-v1/README.md). The source panels have 233 training, 76 calibration and 226 evaluation claim–abstract pairs. The study starts from the same frozen six-layer multilingual MiniLM as the [VitaminC transfer study](../minilm-transfer-scifact-v1/README.md). It makes no weight updates on calibration or test papers.

For a directionally labeled training abstract, enumerate contiguous windows of three sentences. Choose a window containing a complete annotated rationale and one from the same abstract that excludes every annotated rationale sentence. Prefer the largest number of claim terms covered within each group. Skip a directional abstract if either view is unavailable. Assign the original directional label to the first window and a proxy neutral label to the second. Absence of a released rationale does not establish that the second window is insufficient.

The three training arms use the same directional examples, number of negative presentations, batch size, two seeds, four epochs, and trainable final transformer block and classifier. One arm pairs the directional window with a lexical window from an unrelated training claim–abstract pair without released evidence. That source pool is reused when smaller than the number of directional examples. A second arm uses the same-abstract proxy neutral, and the third adds a pairwise margin loss favoring the supported direction in the rationale-bearing window. For every test item use a single reader forward. No gold sentence locations enter the main lexical view.

Freeze the support threshold on the 76-case calibration panel under separate 5% false-support caps for refutation and unknown. Test the same held-out lexical windows for all arms. A separate diagnostic uses the same rationale-present and annotation-excluded construction on held-out directional pairs and reports the fraction getting both raw labels right. This measures recognition of proxy neutrals, with the same caveat about missing annotations.

## RTX 5090 results

Job 42242 found 113 training pairs with both views, including 84 supports and 29 refutations. The unrelated-unknown pool has 92 examples. The 76 calibration cases include 11 refutations and 30 unknowns, allowing zero and one false supports respectively. The held-out lexical panel has 97 supports, 47 refutations and 82 unknowns. The matched diagnostic has 127 directional pairs.

| Training method | Seed | True supports | False support on refutation | False support on unknown | Both diagnostic views correct / 127 |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen | — | 33 | 4 | 4 | 24 |
| Rationale and unrelated unknown | 20261119 | 30 | 3 | 5 | 33 |
| Rationale and same-abstract unknown | 20261119 | 29 | 2 | 5 | 31 |
| Same-abstract pairwise loss | 20261119 | 29 | 2 | 5 | 31 |
| Rationale and unrelated unknown | 20261120 | 30 | 4 | 5 | 33 |
| Rationale and same-abstract unknown | 20261120 | 29 | 2 | 3 | 30 |
| Same-abstract pairwise loss | 20261120 | 29 | 2 | 4 | 31 |

The adaptation improves the constructed paired diagnostic but loses true supports on independently selected lexical windows. The same-abstract variants reduce nominal refutation errors from four to two, with four fewer supports in every seed and objective. The paired 95% interval for support retention change is −8.25 to −1.03 percentage points for those runs. The additional pair loss does not raise lexical support retention relative to classification on the same-abstract data. All models still accept some refutations on evaluation after meeting the small calibration panel's zero-refutation-acceptance rule.

Single-item lexical-window P95 is 1.34–1.41 ms across the seven arms. The benchmark retains the trained parameter subsets and checksum provenance on the compute node. The results distinguish a diagnostic gain from a gain under the actual lexical retrieval policy. They do not demonstrate more supported expert recommendations. [Aggregate results](results.json)

Run with the pinned [SciFact archive and MiniLM checkpoint](../scifact-small-nli-v1/README.md) on a Slurm CUDA node.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.scifact.window_supervision \
    SCIFACT_ARCHIVE MINILM_CHECKPOINT NEW_RESULT NEW_WEIGHTS_DIR
```
