# Criterion-wise expert recall at a fixed verification budget

Use the 25 multi-criterion enterprise development requests in the [source-extent diagnostic](../enterprise-evidence-extent-v1/README.md). The same query reader, 52,667 owned sources, FP32 exact scorer, and frozen verifier apply. Each arm verifies five experts, two owned sources per expert, and the same leading source excerpt length. All six arms score 730 criterion–source pairs. The original arm reproduces the earlier candidate and source order and aggregate support counts.

| Candidate order | Method |
| --- | --- |
| Whole request | Best source match to the complete request per expert |
| Criterion RRF | Keep the first two whole-request experts. Fill the other slots by mean reciprocal ranks of each expert's best source for each criterion. |
| Joint bottleneck | Keep the same two anchors. Fill the other slots by weakest criterion rank, then mean reciprocal rank. |

The criterion projections use the parsed objective and remaining prior qualifications. Qualification projections include the objective for context. Rank every criterion against the same source inventory. Candidate reciprocal ranks use a constant of 60 and competition ranks. Both criterion-based routes use identical projections.

For each candidate route, compare the two best whole-request sources against two criterion-covering sources. The cover heuristic greedily selects sources by the sum of per-criterion best reciprocal source ranks and orders the selected pair by whole-request score before verification. Use the same two-wave verifier on the leading 800 characters of each chosen source. The routing and selection rules are fixed without reference to human labels or verifier outcomes.

## Results on one RTX 5090

| Candidate route | Source route | New positions vs whole request / 125 | Model-supported expert–criterion slots | Experts with all criteria supported | Requests with at least one supported expert |
| --- | --- | ---: | ---: | ---: | ---: |
| Whole request | Original | 0 | 46 | 3 | 2 |
| Whole request | Criterion cover | 0 | 45 | 3 | 2 |
| Criterion RRF | Original | 45 | 49 | 4 | 2 |
| Criterion RRF | Criterion cover | 45 | 48 | 4 | 2 |
| Joint bottleneck | Original | 49 | 48 | 3 | 2 |
| Joint bottleneck | Criterion cover | 49 | 47 | 3 | 2 |

RRF adds one fully supported expert outside the original shortlist, on a request that already had a recommendation. The paired interval for its gain in supported positions per request is 0.00 to 0.12. Joint bottleneck and criterion-covering sources do not increase the number of supported requests. The extra criterion encoding and exact vector score pass have P50 9.87 ms and P95 10.20 ms. Those times exclude ranking, verification and transport. They are an incremental component diagnostic, not end-to-end comparisons.

This is a frozen-model development result on five candidates per request. The two preserved anchors protect original leading candidates and also limit the size of the alternative shortlist. Independent judgments are needed to assess relevance, personal contribution, and the possible verifier errors. The sealed cohort was unused. [Full aggregates](results.json)

## Reproduce

Run the authorized development projections from the linked benchmark within a Slurm CUDA allocation:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_criterion_recall \
  --queries DEVELOPMENT_QUERIES --corpus DEVELOPMENT_SOURCES \
  --models CHECKPOINT_DIRECTORY --output NEW_RESULTS
```

Job 41151 ran the released experiment. Request and source texts are excluded from the release.
