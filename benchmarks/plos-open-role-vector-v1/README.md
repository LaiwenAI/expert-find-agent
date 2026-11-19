# Reuse query vectors before requesting duty repair

The [two-heading repair experiment](../plos-open-role-shortlist-v1) showed that
switching the proposed duty can recover correct matches. This experiment tests
whether a local decision can resolve simpler cases using the query embedding
already computed for role retrieval.

All methods use the same source nomination, source confirmation, applicability
gate, and two-role shortlist. A chosen duty must differ from the explicitly
named duty and share at least one publisher-credited author with it. Structural
feasibility cannot make a losing semantic candidate win.

| Method | Role decision |
| --- | --- |
| Embedding gap | Use the existing margin between the shortlisted definition scores |
| Supervised ridge | Fit a five-output linear head from synthetic role labels |
| Distilled ridge | Fit the same head to the frozen reranker's clipped yes/no scores |
| Supervised then repair | Return the confident supervised decision if it can bind to a credited author. Otherwise request the frozen two-head repair when eligible |

Both ridge heads reuse the same frozen Qwen3-Embedding-0.6B vector. The
[84 synthetic descriptions](../plos-credit-edits-v1/role-training.json)
include 30 descriptions from the five mapped in-inventory roles and 54
descriptions from nine other roles. The frozen Qwen3-Reranker-0.6B teacher
scores the synthetic descriptions against the five legacy headings offline.
The training solve uses a fixed ridge penalty of 0.1. Each local decision
cutoff equals the highest incorrect margin on five older paraphrases and five
outside activities. Normalized training, calibration, and both evaluation
wording banks are disjoint.

## Results

The evaluation searches 197 PLOS articles, with 72 supported queries per clue
and separate withheld-source and outside-activity views. Language examples
repeat across articles. Comparison timing sums the measured component paths
and charges the combined policy for a verifier call only when it would invoke
one. It excludes service queues and transport.

Slurm jobs 42893 and 42895 produced identical quality and dispatch counts. The
table uses timings from job 42895 on an RTX 5090. Each row covers 72
supported requests with the source present.

| Clue | Method | Clean | Wrong duty | Wrong source | Calls | Mean / P95, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Excerpt | Frozen proposal | 18 | 13 | 0 | 0 | 14.88 / 15.33 |
| Excerpt | Two-head repair | 17 | 0 | 0 | 31 | 18.52 / 23.89 |
| Excerpt | Embedding gap | 14 | 0 | 0 | 0 | 14.89 / 15.33 |
| Excerpt | Supervised ridge | 18 | 0 | 0 | 0 | 14.90 / 15.33 |
| Excerpt | Distilled ridge | 14 | 0 | 0 | 0 | 14.89 / 15.33 |
| Excerpt | Supervised then repair | 31 | 0 | 0 | 13 | 16.48 / 23.91 |
| Keywords | Frozen proposal | 14 | 9 | 1 | 0 | 14.37 / 14.63 |
| Keywords | Two-head repair | 14 | 0 | 0 | 23 | 17.03 / 23.28 |
| Keywords | Embedding gap | 9 | 0 | 0 | 0 | 14.38 / 14.64 |
| Keywords | Supervised ridge | 14 | 0 | 0 | 0 | 14.38 / 14.64 |
| Keywords | Distilled ridge | 9 | 0 | 0 | 0 | 14.38 / 14.63 |
| Keywords | Supervised then repair | 23 | 0 | 0 | 9 | 15.48 / 23.24 |

The local supervised head keeps the original proposal's clean coverage on
these examples and rejects its duty errors, but makes no successful role
switches. The distilled head loses coverage relative to the supervised head
under the same cutoff procedure. The combined policy recovers every
structurally reachable supported request that passed both fixed upstream gates
on this catalog. It also keeps one wrong-source response when the intended
article is withheld in the keyword view, whereas two-head repair alone
abstains. All methods abstain on the original outside-activity views.

A second view uses the earlier
[confusable wording bank](../plos-credit-role-transfer-hard-v1/phrases.json),
which includes exclusions and incidental mentions of nearby roles. Source
clues, intended duties, and publisher attribution labels stay fixed. Neither
the local heads nor the reranker are dispatched for any of the 72 supported
requests per clue in this view. All arms return zero. This is a transfer
failure of the complete operating policy and does not isolate the local head's
classification accuracy in isolation.

The combined policy reduces calls on the original wording and saves mean
component latency. P95 still includes a model call. The confusable variant and
withheld-source error constrain this evidence to the tested operating regime.

[Aggregate results](results.json)

## Reproduce

Use the command from the [shortlist benchmark](../plos-open-role-shortlist-v1),
a fresh output path, and the additional inputs:

```sh
--student-training benchmarks/plos-credit-edits-v1/role-training.json \
--repair-reference benchmarks/plos-open-role-shortlist-v1/results.json \
--stress-phrases benchmarks/plos-credit-role-transfer-hard-v1/phrases.json
```
