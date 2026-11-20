# Published MERIT assessor with generation and direct labels

Evaluate the released MERIT-4B reviewer assessor using the
same 129 held pairs (45 positive, 60 components) as the
[compact profile benchmark](../merit-compact-source-head-v1).
A positive label is a reviewer self-rating of four or five.
Use the publisher's full evaluation prompt, response cap of
4,096 tokens, left padding, greedy generation and original
fallback parsing. Run batch size 8 rather than the author's
default 16 on one RTX 5090. All held prompts fit the 5,120-token
cap. See the [MERIT paper](https://aclanthology.org/2026.emnlp-main.2227/)
and [released checkpoint](https://huggingface.co/Luli3220/MERIT-4B-reviewer-assessor).

With identical weights, score next-token logits for the two
labels after a `[FINAL_LABEL]` prefill. Compare the original
full prompt, a shortened instruction with the full profile,
and the shortened instruction with two lexically selected
complementary papers. The direct method emits a label only.
Fit thresholds on the separate 125-pair calibration partition
using serial scores. Report unadjusted, balanced and high-recall
thresholds. Repeat scoring in batches of eight with the same
frozen serial thresholds. Matched batch timing covers tokenization
through decision for prebuilt prompts, with the same sample
order and batch sizes as generation. It excludes model loading,
prompt construction, full-catalog retrieval and serving.

## Results

Slurm jobs `45102` and `45110` both completed on the same node.
Five of 129 generation sequences hit the cap without EOS. Five outputs
were missing a parsed label and were treated as negatives under the
author's protocol. The held numbers below use the balanced
calibration thresholds for direct routes.

| Same 4B checkpoint | TP / 45, batch 8 | FP / 84, batch 8 | Balanced accuracy | Batch P95 | Whole held set |
| --- | ---: | ---: | ---: | ---: | ---: |
| Author generation | 35 | 29 | 71.63% | 158.15 s | 1,454.45 s |
| Original full prompt, direct label | 36 | 30 | 72.14% | 3.77 s | 46.41 s |
| Short instruction, full profile | 31 | 37 | 62.42% | 3.33 s | 40.02 s |
| Short instruction, two papers | 23 | 23 | 61.87% | 2.11 s | 26.54 s |

The original-prompt direct label cuts batch P95 by about 42x
and time over the held panel by about 31x. At its calibrated
threshold the serial form has 35 true and 32 false positives,
with a P95 of 263.67 ms. Grouping into batches changes five
labels. The unadjusted direct form predicts positive for every
pair, so threshold calibration is essential. The shortened
prompt reduces balanced accuracy across both profile depths.

The separate 0.6B cross-encoder on this cohort returns 31 true
and 19 false positives at its balanced threshold. The dense
best-source route returns 36 true and 27 false positives at its
high-recall calibration threshold. These are alternative
operating points. The reports hold aggregate confusion counts
for author generation, so differences against generation have
no paired uncertainty estimate. They do not establish statistical
noninferiority, citation quality, or enterprise suitability.
[Aggregate results](results.json).

## Reproduce

Check out the public MERIT test and prompt release at
`5fad0ea06148c3c8817aac841de46c98a75eb40c`. Download the model
repository at the revision recorded in the results. On a Slurm
CUDA allocation, run the following commands with new output
paths.

```sh
export PYTHONPATH=src:.
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
python -m experiments.decision.merit_author_assessor \
  MERIT_TEST_JSON MERIT_PROMPTS_PY MERIT_MODEL_PATH NEW_GENERATION_RESULT --batch-size 8
python -m experiments.decision.merit_author_direct \
  MERIT_TEST_JSON MERIT_PROMPTS_PY MERIT_MODEL_PATH NEW_DIRECT_RESULT --matched-batch-size 8
```
