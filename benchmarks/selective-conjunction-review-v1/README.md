# Verify only selected conjunctions against original contribution notes

Reuse the [full-proof calibration](../engineering-proof-group-calibration-v1).
Index-only and all-fusion outcomes at the original frozen thresholds
must reproduce. Fit the routing rule exclusively on the 30 eLife
development groups. Measure on 30 disjoint held eLife groups and the
four previously explored Sensors source pairs. Each request fixes
both sources, contributor linkage and the condition scope.

Review replaces the entire index-only candidate set with the fused
candidate set for that request. Score the following priorities before
calling the original-text reranker:

| Route | Priority |
| --- | --- |
| Hash | Fixed pseudorandom order by request identity and text |
| Margin | Closest candidate proof score to the index issuance cutoff |
| Disagreement | Number of authors on whom the ridge and definition indexes disagree about issuance |
| Learned gain | Group-weighted ridge model of the change in correct issuance plus twice the improvement in unsupported issuance |

The learned rule uses the best ridge and definition proof margins,
nearest cutoff distance, bank disagreement, issued and total owner
counts, and mean difference between the two side scores. Reuse the
role banks and the proof thresholds. Fit a definition-only threshold
under the original group rule on development for computing cheap
disagreement. The learned route requires no held labels or reranker
scores at routing time. Request hashes break all ties. The loss
weights unsupported-request repair twice as heavily as gaining a
correct request. They were not optimized on the held sets.

## Budget and component replay

Stratify by the number of candidate clause rows. In each stratum,
review the highest-priority integer quarter of requests under each
policy. This is a retrospective matched-work comparison that needs
the held cohort's priority distribution and uses none of its labels.
Its exact request and verifier-check totals match across the four
rules.

Separately freeze the corresponding score and tie-breaking cutoff
for each stratum on development. Test that policy per request without
consulting other test requests. Strata absent from the learned
budget receive no original-text review. Run index only, always
fused, frozen margin, disagreement and learned gain sequentially on GPU.
A follow-up arm chosen after the first timed pilot uses the same
disagreement cutoff, reviews only requests with an indexed result,
and only withdraws indexed authors if fusion rejects them. It cannot
add a newly recommended author by construction. Timing covers phrase encoding, role scoring, routing, any reranking,
conjunction and serialization, using resident source context and
author membership. Context assembly, source search, transport and
queueing are outside the timer. Rotate policy order and reverse
request traversal in the second task.

## Results

The full routes reproduce the archived decisions in all panels. At
matched work, each selective route checks 44 eLife requests and
132 clause rows, or seven Sensors requests and 14 clause rows.
Cells report correct positive requests and requests with any wrong
issued author.

| Route | Held eLife correct / wrong | Sensors correct / wrong |
| --- | ---: | ---: |
| Index only | 83 / 1 | 8 / 4 |
| Always fused | 80 / 1 | 7 / 2 |
| Hash | 83 / 1 | 8 / 4 |
| Margin | 80 / 1 | 7 / 2 |
| Disagreement | 83 / 1 | 8 / 3 |
| Learned gain | 83 / 1 | 8 / 4 |

Learned gain gives no improvement over the hash control in either
held panel. Margin matches full fusion's counts on Sensors with a
lower fixed work budget, while losing three correct eLife requests.
Disagreement keeps all eight Sensors correct requests while
suppressing one wrong result in this retrospective budget control.
The Sensors differences are measurements on four reused source
pairs.

### Frozen per-request replay

Each replay covers 178 held eLife requests or 30 Sensors requests.
Numbers in the work columns are actual original-text clause checks.

| Route | eLife correct / wrong | Checks | Local P95 (ms) | Sensors correct / wrong | Checks | Local P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Index only | 83 / 1 | 0 | 10.78–10.88 | 8 / 4 | 0 | 10.65–10.77 |
| Margin | 82 / 1 | 72 | 20.68–21.46 | 7 / 2 | 32 | 20.78–21.14 |
| Disagreement | 82 / 1 | 92 | 21.46–21.52 | 8 / 3 | 8 | 20.36–20.61 |
| Disagreement veto | 82 / 1 | 88 | 21.31–21.42 | 8 / 3 | 8 | 20.43–20.57 |
| Learned gain | 83 / 1 | 152 | 21.47–21.57 | 8 / 4 | 2 | 10.77–10.84 |
| Always fused | 80 / 1 | 548 | 24.10–24.20 | 7 / 2 | 60 | 20.85–20.95 |

The veto and replacement rules issue identical held outputs in
these runs. The veto saves one eLife request with four rows and
has identical Sensors work. The Sensors veto removes one unsupported
request and reduces affected groups from two to one relative to the
index. Its P95 remains close to fusion despite reducing raw checks
from 60 to eight, because four of the 30 requests still take the
longer path. On held eLife, the same policy loses one correct
request with unchanged wrong issuance. The index remains the
stronger eLife operating point. The learned router does not improve
the index on either panel.

The cohort-matched budget operates on frozen batched scores; online
batching can change routing counts close to the cutoffs. Timed
quality and check counts are from the actual per-request decisions.
Both tasks of Slurm array 46999 on ws1 reproduce every answer
signature and work count. Earlier score and policy results match
the pilot in array 46994. [Aggregate results](results.json).

## Reproduce

Restore the inputs from the
[proof-group calibration](../engineering-proof-group-calibration-v1)
and [original-text audit](../engineering-raw-conjunction-v1). Use a Slurm
GPU and a fresh results path.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.selective_conjunction_review \
  ENGINEERING_PANEL LINKED_XML_DIRECTORY CROSS_SOURCE_PANEL \
  benchmarks/engineering-cross-source-search-v1/source-manifest.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  QWEN_EMBEDDING QWEN_RERANKER \
  benchmarks/engineering-proof-group-calibration-v1/frozen-pairs.json RAW_NOTES \
  benchmarks/engineering-raw-conjunction-v1/results.json \
  benchmarks/engineering-proof-group-calibration-v1/results.json NEW_RESULTS
```

Repeat with `--reverse` and a separate output path.
