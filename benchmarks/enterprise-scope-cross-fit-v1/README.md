# Cross-fit the one-pass qualification parser on real request layouts

The [synthetic-only experiment](../enterprise-bidirectional-scope-v1)
found that a one-pass open-weight student was inexpensive, but all
80 enterprise development requests fell back to the 4B parser.
Here the supervision is varied while keeping the bidirectional
Qwen3-Embedding-0.6B backbone, clause anchors, rank-eight adapters,
and fitting budget fixed. The independent scope review remains
pending. Qualification labels on the enterprise development cohort
come from the frozen Qwen3-4B-Instruct-2507 parser.

Freeze five folds before running either model. Join development
rows to their original workbook and metadata by exact text and
provenance. Link identical normalized issuers, identical normalized
bodies, and near-duplicate character four-grams with the existing
panel thresholds. Assign the connected components whole to folds.
For each test fold use the next fold for gate calibration and the
other three for training. Test texts and labels are never used to
fit or calibrate that fold's student. These are reused development
requests with within-study cross-fitting, and the split does not
certify independence from every historical experiment or paraphrase.

| Training route | 192 document exposures per epoch |
| --- | --- |
| Synthetic | All 192 authored mixed-scope documents |
| Teacher | The 48 training-fold requests, repeated four times, with all clauses teacher-labeled |
| Mixed | The same 48 real requests repeated twice and 96 synthetic documents sampled without replacement |

Use six epochs, one seed, fixed model updates, and positive weights
equal to the negative-to-positive clause ratio capped at 30. The
document and step counts are matched. Annotation density and unique
document counts differ. The comparison is an intervention on
supervision source, including the supervision density available in
each source.

All routes calibrate on the same 16 real requests of each fold.
A request's confidence is the smallest absolute clause logit.
The cutoff is the largest confidence of any calibration request
with a qualification-set disagreement. Accept only above it.
With no disagreement the cutoff is zero. This empirical rule is
not a coverage or risk guarantee. The test reports qualification
set agreement, accepted disagreements, and whether the gate
reaches requests on which the 4B parser takes over 500 ms.
The all-negative control identifies how many requests match the
teacher by predicting no qualification. Report the authored
heading judgments separately to detect regression against public
gold labels. Each fit sees the same 72 controls in evaluation only.

P95 includes tokenization, student scoring, and separately measured
4B parse time on requests that fall back. This is an additive serial
estimate. Retrieval, recommendation changes, verifier traffic,
concurrency, and client transport are outside this measurement.

## Results

Both RTX 4090 tasks in Slurm array 47442 on ws6 reproduce the
teacher labels, training losses and decisions in reversed fold
and arm order. The 80 requests form 71 groups. All five folds
have 16 requests, with 14–15 groups per fold. The reference
marks 51 of 1,452 clauses as prior qualifications on 25 requests.

| Training route | Reference marks recovered / 51 | Additional marks absent from reference | Exact request sets / 80 | Skips / 80 | Changed sets among skips | P95 with fallback, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic | 0 | 0 | 55 | 17 | 7 | 1,308–1,313 |
| Teacher | 24 | 304 | 43 | 16 | 4 | 1,307–1,313 |
| Mixed | 5 | 161 | 46 | 21 | 5 | 1,308–1,313 |

The all-negative control agrees with the reference on 55 requests.
Teacher supervision recovers 24 clause marks in held groups but
adds many others, and request-level agreement declines. On the
separate public heading controls, the five teacher-only fits
correctly classify both views of zero pairs out of 36 per fit.
The synthetic fits get two pairs each, and mixed fits get zero to
two pairs. These are authored controls and do not adjudicate the
teacher's decisions on real requests.

The real-request calibration permits 16–21 skips, of which 4–7
change the qualification set. Only one or two skipped requests per
arm took over 500 ms in the reference. The reference P95 is
1,289–1,294 ms, and all gates slightly increase this tail estimate.
Median estimates are lower for the teacher and mixed routes. The
measured agreement and tail latency do not support replacing the
4B path. A larger source of independent clause judgments should
separate genuine model mistakes from teacher disagreements
before further calibration. The existing frozen qualification
forms can be completed using the [offline review page](../enterprise-qualification-scope-audit-v1/offline-review.html).

[Aggregate runs](results.json).

## Reproduce

Freeze the private groups with the restricted original workbook and
development metadata from the known fresh-evaluation projection.
Share only aggregate counts and digests.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_scope_folds \
  DEVELOPMENT_QUERIES FRESH_METADATA ORIGINAL_WORKBOOK NEW_PRIVATE_FOLDS.json
```

Then run in a Slurm CUDA allocation with the checkpoints from the
[synthetic-only experiment](../enterprise-bidirectional-scope-v1),
and repeat with `--reverse` and new output.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_scope_cross_fit \
  DEVELOPMENT_QUERIES benchmarks/enterprise-heading-roles-v1/controls.json \
  benchmarks/enterprise-scope-transfer-v1/controls.json PRIVATE_FOLDS.json \
  QWEN3_EMBEDDING_CHECKPOINT QWEN3_INSTRUCT_CHECKPOINT NEW_AGGREGATE.json
```
