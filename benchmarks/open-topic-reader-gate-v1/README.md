# Route the six-input source reader under a call-rate target

The [source-reader experiment](../open-topic-plan-reader-v1)
found unstable gains from semantic scoring at high local latency.
This study asks whether an inexpensive signal can assign those
calls to requests most likely to gain a tagged proof. Every
request still reads exactly three declarations. The candidate
pool, role index, per-request reader input limit and output rule
are those of the archived study. Require both reader and rank
outputs to match the archived signature before comparing routes.

Extract features before any source has been read. Count how many
compatible pairs are fully contained in the joint plan and how many
remain outside, count the candidate source sets, and record the
best worst citation rank for each group and their signed and
absolute differences. Require at least two compatible orientations
and more than one set before considering a model call. Verify that
this eligibility condition reproduces the observed reader executions.

| Gate | Priority score |
| --- | --- |
| Stable | Fixed request hash, used as a label-blind control |
| Weak baseline | Worse retrieval rank for the weakest source of the best baseline proof |
| Small gap | Smaller absolute rank gap between the best baseline proof and its strongest alternative |
| Ridge | Standardized features with fixed ridge penalty 8. Predict signed success change for the six-input reader from paired training requests. |

Fit and calibrate on PLOS in five folds by hashed topic pair.
The outcome for fitting is the change in publication-tag-supported
proofs. Untagged outputs count as zero in the fitted signal; the
separate missing-judgment bounds permit either true label. Set each
threshold to call on 10% of the calibration cohort, using only
training data in every fold. Transfer the threshold and, where
applicable, weights from all PLOS to eLife. The 10% is a calibration
target. Actual calls on held-out queries can differ. eLife contributes
no labels to training or threshold selection.

Replay the routed decisions using the measured arm times, adding
the measured cost of computing gate features. This estimates local
P95 without a live service or queue. Contrast these estimates with
the original joint path and with the six-input reader. Also compute
an optimistic labeled-gain ceiling at an exact 10% call budget by
giving a hindsight oracle the source-reader gains in the test panel.
That ceiling describes policy headroom under the observed labels,
without assuming the gains are predictable. Neither panel contains
independently judged expert suitability.

## Results

Both tasks of Slurm array 47516 ran on RTX 4090 GPUs on ws6
and reproduced every discrete result after reversing execution.
There are 12 label gains and 10 losses when replacing joint
planning with the six-input reader wherever eligible on PLOS,
and one gain and two losses on eLife.

| Gate | PLOS calls / 160 | PLOS gains / losses | PLOS estimated P95 | eLife calls / 51 | eLife gains / losses | eLife estimated P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stable | 13 | 2 / 2 | 55–56 ms | 17 | 0 / 2 | 56–58 ms |
| Weak baseline | 17 | 1 / 3 | 47–50 ms | 26 | 0 / 2 | 56–58 ms |
| Small gap | 13 | 1 / 0 | 53–55 ms | 5 | 0 / 2 | 56–57 ms |
| Ridge | 25 | 3 / 3 | 51–54 ms | 11 | 1 / 2 | 52–53 ms |

Role joint alone takes 22–23 ms on PLOS and 21–22 ms on
eLife. Here even the low realized call rates are enough to
raise the 95th percentile. At the calibration target, the
hindsight 16-call and five-call ceilings are +12 and +1
tagged proofs, respectively. The operational gates do not
approach those ceilings and can miss the call-rate target
after transfer. The ridge gate changes neither the 85/140
PLOS tagged outcomes nor the equal-issuance count. Its eLife
transfer yields 45 rather than 46 supported proofs at 49
issuances. Funding the rare reader gains needs a better
predictor and independent verification of the changed cases.
Keep the reader-free joint plan as the first-query default.

[Aggregate results and replication hashes](results.json).

## Reproduce

Use the [same frozen inputs](../open-topic-plan-reader-v1#reproduce)
and a fresh Slurm CUDA allocation. Repeat with `--reverse`.
The gate does not use CSFCube inputs for training.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.open_topic_reader_gate \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json QWEN_EMBEDDING \
  benchmarks/mixed-publisher-joint-v1/results.json \
  benchmarks/live-joint-source-request-v1/results.json \
  benchmarks/open-topic-plan-reader-v1/results.json QWEN_RERANKER NEW_RESULTS
```
