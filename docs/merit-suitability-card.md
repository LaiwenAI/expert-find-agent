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

## LR-Bench pairwise retrieval

This separate benchmark uses the two original [LR-Bench pairwise files](https://huggingface.co/datasets/Gnociew/LR-bench/tree/6b950b4a6f309d4e6fd3749744d2ddd9005fa6f8). For every row, the anchor ID, both alternative IDs, comparison type and reference ratings match the [RATE release predictions](https://github.com/Gnociew/RATE-Reviewer-Assignment/tree/8b85915315a5ebca92cda8a0d901ecaaeb8c0cba/predictions). The released profiles and candidates are fixed. No scored pairs contain an exact target title-and-abstract match in either supplied profile. This check does not establish freedom from near-duplicates or pretraining overlap.

| Method | Paper ranks reviewers, accuracy / loss | Reviewer ranks papers, accuracy / loss |
| --- | ---: | ---: |
| BM25 max | 64.62% / 0.2806 | 68.75% / 0.2636 |
| Frozen Qwen3-Embedding-0.6B max | 66.51% / 0.2620 | 74.32% / 0.2041 |
| Frozen Qwen3-Embedding-0.6B top-three mean | 67.45% / 0.2580 | 75.93% / 0.1889 |
| RATE author predictions | 76.89% / 0.1795 | 75.25% / 0.1926 |

The [official evaluator](https://github.com/Gnociew/RATE-Reviewer-Assignment/blob/8b85915315a5ebca92cda8a0d901ecaaeb8c0cba/evaluation_script.py) reports strict pairwise accuracy and rating-gap-weighted Kendall loss. The paper-centric split contains 212 pairs and 150 anchors. The reviewer-centric split contains 1,184 pairs and 187 anchors. A paired bootstrap by anchor gives a top-three minus RATE accuracy difference of -9.43 percentage points (95% interval -16.53 to -1.88) on paper-centric comparisons and +0.68 points (-2.35 to +3.59) on reviewer-centric comparisons. The directions should be kept separate. Paper-centric comparisons are closer to finding people for a request and should be the primary LR-Bench measure. Reviewer-centric results do not establish equivalence. Both splits concern self-rated familiarity.

The frozen model embeds titles and abstracts with a 512-token limit and reuses the profile vectors. Slurm job `40205` on `ws5` (RTX 5090) embedded 1,493 unique profile texts in 6.26 s and 689 distinct targets in 3.25 s at batch size 32. Cached scoring and aggregation for 896 unique target/profile combinations took 8.18 ms in total. These are batch component measurements. The released RATE predictions have no comparable serving timing.

To reproduce, save `evaluations_pc.json` and `evaluations_rc.json` from the pinned dataset above, plus `RATE_pc.json`, `RATE_rc.json`, and `evaluation_script.py` (renamed `rate-evaluation_script.py`) from the pinned RATE release into the same directory. The runner verifies their hashes. Use the official frozen Qwen revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` and its verified checkpoint provenance file.

```bash
PYTHONPATH=src:. python -m experiments.retrieval.lrbench_frozen_baselines \
  --release artifacts/lrbench-release-6b950b4 \
  --output artifacts/lrbench-frozen-v1
```

### Matched-size profile comparison

A fixed 2 × 2 comparison uses frozen Qwen3-Embedding-0.6B and the [RATE-0.6B checkpoint](https://huggingface.co/Gnociew/RATE-0.6B/tree/3afb29314108e4fdf834e48a7b5b30fa64d9afe7). Both use BF16, mean pooling over non-padding tokens, and the RATE paper prompt. The compact arm embeds one author profile from the released keywords. The raw arm averages the three highest paper-to-paper cosine scores, using the paper prompt for profile sources as well. Token IDs match across checkpoints for all 2,447 distinct inputs, with a maximum length of 823 and no truncation. The [RATE paper](https://aclanthology.org/2026.acl-long.2035/) describes training on top of Qwen3-Embedding.

| Checkpoint and profile | Paper ranks reviewers, accuracy / loss | Reviewer ranks papers, accuracy / loss |
| --- | ---: | ---: |
| Frozen, raw top-three | 70.28% / 0.2394 | 75.59% / 0.1895 |
| Frozen, compact keywords | 73.11% / 0.2247 | 73.56% / 0.2145 |
| RATE-0.6B, raw top-three | 70.75% / 0.2287 | 75.59% / 0.1858 |
| RATE-0.6B, compact keywords | 75.47% / 0.1955 | 74.32% / 0.2035 |

On the paper-centric split, the compact-minus-raw accuracy difference was +2.83 pp (anchor bootstrap 95% interval -3.80 to +8.99) with the frozen checkpoint and +4.72 pp (-1.42 to +10.35) with RATE-0.6B. Changing only the checkpoint with compact profiles gave +2.36 pp (-0.50 to +5.34). Compact profiles are a candidate for first-pass ranking. These exploratory comparisons do not establish a quality gain or equivalence. Both models lose point-estimate accuracy in the reverse direction with compact profiles.

The release contains 276 distinct compact profiles versus 1,493 distinct raw-history texts. In Slurm job `40216` on an RTX 5090, RATE-0.6B took 2.24 s to encode compact profiles versus 6.80 s for raw history offline. On the same 896 candidate scores, cached dot products took 1.40 ms versus 5.26 ms for raw aggregation, while batched target encoding took 3.34 s. These measurements exclude index lookup and service overhead. Repeat job `40212` produced identical prediction files. The next validation needs matched candidate recall on an independent paper-centric collection and an explicit trace from compact scores back to source-level evidence.

Reproduce using the assets from the preceding section plus `data/keywords.json` and `scripts/RATE.py` (save as `rate-RATE.py`) from RATE commit `8b85915315a5ebca92cda8a0d901ecaaeb8c0cba`. Download `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `model.safetensors` from the pinned RATE-0.6B revision and place them under `artifacts/models/rate-0.6b`. Write `checkpoint-provenance.json` in the same format as the frozen checkpoint, with the RATE repository, revision, and SHA-256 of each file. The runner verifies upstream inputs and checkpoint hashes.

```bash
PYTHONPATH=src:. python -m experiments.retrieval.lrbench_matched_profiles \
  --release artifacts/lrbench-release-6b950b4 \
  --output artifacts/lrbench-matched-profiles-v2
```
