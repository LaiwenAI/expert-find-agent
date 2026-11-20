# Verify opposite scientific evidence requirements on the same abstract

Test transfer of mandatory-criterion verification from news to
scientific abstracts. A claim–abstract pair is evaluated under both
a support requirement and a contradiction requirement. The original
human SciFact-Open stance label determines which view is correct.
The input wording for the two views differs by the required stance.
An unannotated abstract is never promoted to a negative example.

The [IFIR scientific subset](https://aclanthology.org/2025.naacl-long.511/)
motivates the support/contradiction task. Its released relevance
table does not explicitly label negatives. Use the 45 distinct
claims in its scientific `test_data.json` as the calibration group
and the remaining annotated claims in
[SciFact-Open](https://aclanthology.org/2022.findings-emnlp.347/)
as the held-out group. Discard test pairs if their abstract ID was
seen in calibration. This removes two pairs and leaves 195 test
pairs across 160 claims, with 102 human support labels and 93 human
contradiction labels. The two groups have disjoint claim and
abstract IDs. This is a new split for the stated experiment on a
previously studied public dataset.

## Fixed evidence and verifier arms

Each arm receives the same title and abstract prefix of up to 1,800
characters. No human rationale locations or stance labels are
included in the model input.

- **Separate binary:** two frozen Qwen3-Reranker-0.6B scores, one for
  each required stance. Decide each request from its own score.
- **Contrastive binary:** reuse the same scores. For each request
  subtract the opposite-stance score. Both scores would have to
  be computed for a real request. Apply a nonnegative minimum
  cutoff to prevent approving both stance requests.
- **Ternary reader:** one Qwen3-4B-Instruct-2507 forward with explicit
  choices for support, contradiction and insufficient evidence. Score
  a requested stance by its margin over the stronger alternative.
  Apply a nonnegative minimum cutoff.

Choose a cutoff per stance and route on calibration, with at most
10% observed acceptance of the human opposite stance. Evaluate
unchanged cutoffs and the fixed zero cutoff on the held-out pairs.
Report correct and opposite decisions, simultaneous approval, and
abstention on both requests. The held-out examples have an annotated
stance. They do not estimate the real prevalence of missing
evidence. This benchmark isolates evidence verification and
does not measure expert attribution or end-to-end search latency.

## Results on RTX 5090

Both tasks of Slurm array `44728` on `ws5` reproduced all thresholds
and counts. Held-out results count the same 195 claim–abstract pairs under
both requested relations. The four columns are disjoint.

| Route | Correct relation only | Opposite relation only | Both approved | Neither approved |
| --- | ---: | ---: | ---: | ---: |
| Separate binary, cutoff 0 | 6 | 7 | 157 | 25 |
| Separate binary, calibrated | 5 | 5 | 59 | 126 |
| Contrastive binary, cutoff 0 | 104 | 89 | 0 | 2 |
| Contrastive binary, calibrated | 32 | 24 | 0 | 139 |
| Ternary reader, cutoff 0 | 117 | 44 | 0 | 34 |
| Ternary reader, calibrated | 99 | 23 | 0 | 73 |

The standalone binary scores are often simultaneously positive.
Forcing mutual exclusion removes dual approvals, but at cutoff
zero the compact model keeps only 4 of 93 contradictions and
accepts the opposite support request on 88 of those same 93
cases. The ternary reader with calibrated thresholds instead
recovers 44 of 102 supports and 55 of 93 contradictions. Its 23
opposite approvals include 16 of 102 human supports incorrectly
returned as contradictions, so the 10% calibration cap does not
transfer to that class. Both readers see the same truncated
abstract; no binary input exceeds the model context after that
application-level truncation.

The contrast identifies a weakness in distinguishing requested
evidence polarity, beyond deciding whether a document discusses
the claim. The compact exclusive score is a useful ablation.
The ternary reader is a stronger verification control for testing
when added verification cost is justified. This comparison changes
both the instruction format and the model, so it does not isolate
the contribution of either change.

[Aggregate results](results.json).

## Reproduce

Use the official SciFact-Open release archive and the
[author-published IFIR repository snapshot](https://huggingface.co/datasets/songtingyu/IFIR/tree/0e1bd15954eed1414023961a9f79038cf2aad2c6/scifact_open).
The preparation command checks both hashes. Keep the abstract
projection outside Git.

```sh
PYTHONPATH=src:. python -m experiments.scifact.polarity_gate prepare \
  SCIFACT_OPEN_ARCHIVE IFIR_SCIFACT_TEST_DATA PROJECTION
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.scifact.polarity_gate run \
  PROJECTION QWEN_RERANKER QWEN_4B_READER NEW_RESULTS
```

Run inference on a Slurm GPU. Use `--order-offset 1` for the second
run to reverse panel order.
