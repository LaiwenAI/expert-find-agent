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

## Exploratory selective replay

A post-hoc simulation uses the stored native-choice probability scores to decide whether to escalate to the stored full-reasoning result. Exact target-title hashes partition the same previously examined 610 pairs into 290 development and 320 test pairs. On development, a fixed threshold grid is selected to maximize accuracy subject to simulated mean time at most 70% of full reasoning. The chosen maximum-choice-probability threshold is 0.75. On the test partition, 141/320 cases escalate; accuracy is 76.88%, compared with 75.00% for native choice and 76.25% for full reasoning. The paired difference from full is +0.63 percentage points, with an exact-paper-cluster bootstrap interval of [-0.98, 2.44] points; this is not proof of equivalence. Simulated serial time is 2.54 seconds versus 3.80 seconds for full reasoning alone, using separately observed request durations from the earlier concurrent and interleaved run. This is not an actual cascade measurement, and test labels have appeared in earlier overall comparisons. Confidence is an uncalibrated model score; self-rated reviewer expertise does not establish enterprise applicability. Reproduce locally with `PYTHONPATH=src python -m experiments.analyze_merit_selective_replay`.

A subsequent fresh-call pilot fixes the 0.75 threshold and selects 40 distinct target-title groups from the replay test partition by a label-blind hash. The paired arms alternate order with three workers and no retries. All 80 arm attempts succeed; the cascade escalates 19 cases and matches self-ratings on 35/40, versus 34/40 for full reasoning. Mean measured arm times are 3.06 and 5.14 seconds respectively, including both serial calls on escalation but excluding client initialization. These requests are new but the same source cases and overall benchmark were previously studied. Forty pairs are insufficient to establish equivalence or a general latency distribution; no industrial expert-fit conclusion follows. Run `PYTHONPATH=src python -m experiments.merit_selective_live_pilot` to create a new local pilot directory; the script refuses to overwrite an existing run.

A separate integration pilot uses the reusable `selective_decision` component with the same fixed threshold on 12 further distinct title groups, excluding all 40 initial pilot groups. All 24 arm attempts succeed, and four component calls escalate. Each arm agrees with 10/12 self-ratings; measured mean times are 2.35 seconds for the cascade and 4.16 seconds for full reasoning. These are new calls but the overall 610-pair source was previously examined. The pilot verifies connectivity and routing, not equivalence, robust latency, or enterprise suitability. Reproduce with `PYTHONPATH=src python -m experiments.merit_selective_live_pilot --component`; the original pilot protocol is required to exclude its groups.
