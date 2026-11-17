# MERIT Public Expertise Matching Test Adaptation

This adaptation is only for **paper-reviewer expertise matching** and cannot replace labels for enterprise requirement transfer or project success. [MERIT (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.2227/) states in Section 4.1 that LR-Bench expertise is based on human self-ratings on a 1-5 scale, with 4/5 as positive and the rest as negative; this adaptation fixes the authors' publicly released 610 for testing rather than model-generated silver labels.

The official code is pinned to commit `5fad0ea06148c3c8817aac841de46c98a75eb40c`, sourced from the [MERIT repository](https://github.com/Luli3220/MERIT). A selective download of 9 files totaling 20,021,659 bytes was verified item by item against Git blob SHA-1, file length, and SHA-256; see `docs/exploration/merit-release-assets-20261117.json` for the manifest. The data and author code are stored in `data/public/merit-release-20261117` and are not redistributed with this repository's MIT source package.

610 contains 254 positive and 356 negative examples, corresponding to 509 distinct target papers after title normalization. The author materials provide only a list of papers and therefore cannot guarantee reviewer identity independence; exact title grouping also does not amount to semantic near-duplicate exclusion. The authors' human self-ratings are independent of locally generated same-model silver labels, but remain self-rating standards for a specific task and do not prove capability on future enterprise projects.

## Frozen Comparison

All arms use `gpt-6-luna`. No models are trained, and the MERIT-trained 4B/8B weights are not run:

1. `official-long`: directly uses the complete system/user prompts released by the authors and generates requirements, evidence, a summary, and a final label, with a cap of 4096 completion tokens.
2. `short-chat`: retains the same decision requirement for the core plus supplementary skills and all target/material text, returning only a short label under a strict schema, with a cap of 128 completion tokens.
3. `native-choice`: uses the same short decision requirement and complete materials, returning a binary result via native Decisions.

This is a comparison of the actual prompts, generation volumes, and interface combinations. It cannot be called an exact reproduction of the MERIT training method, nor is it an experiment in which the internal computations of the two interfaces are completely identical. The six call permutations are cycled in original 610 order, with three concurrent calls and no retries. The target title/abstract/introduction and all candidate paper titles/abstracts are retained; the 5K prompt truncation/filtering in the authors' local evaluation script is not carried over. Scores are read for evaluation only after inference; inference cases/payloads contain neither `score` nor human labels.

## Failures and Evaluation

API errors, missing final fields, and format mismatches remain `label=null`; no 0 is imputed. The analyzer must check that all 610 x 3 terminal files exist and refuse to output an accuracy report if any are missing. The report gives precision/recall/F1/accuracy on valid cases, coverage, and the worst-/best-case recognition interval for the full queue including failures; the interval is not a confidence interval. Comparisons are also bootstrapped by target-paper title on commonly valid paired cases. Latency uses all actual attempts, including waiting time for errors. Grouping cannot account for unknown reviewer-identity correlations. It cannot be guaranteed that pretraining did not include these public materials or labels.

```bash
python experiments/fetch_merit_release_assets.py
PYTHONPATH=src python experiments/merit_suitability_comparison.py --prepare-only
PYTHONPATH=src python experiments/merit_suitability_comparison.py
PYTHONPATH=src python experiments/analyze_merit_suitability.py
```

The inference command makes paid API calls; existing results are reused when the frozen input digest matches. All 610 cases and 1830 requests completed with valid results: accuracy for full inference/short Chat/native Decisions was 73.28%/68.69%/71.15%, respectively, and mean request times were 3.84/1.64/0.91 seconds. The accuracy difference for native Decisions relative to full inference was -2.13 percentage points, with a paper-grouped bootstrap interval of [-5.51,1.32]; crossing zero does not prove equivalence. This is a cost-quality trade-off and cannot be described as an accuracy improvement. The 20-item sealed set from the original enterprise 100 requirements was not used in this public auxiliary task.
