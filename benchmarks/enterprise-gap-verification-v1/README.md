# Verify only outstanding source criteria

Keep the same 80 enterprise development requests, 52,667 sources and verified open Qwen checkpoints as the [qualification-boundary benchmark](../enterprise-qualification-guard-v1/README.md). Use qualification-boundary parsing, live exact dense retrieval, five candidates and up to two sources per owner. Every arm cites the first source with a positive verdict for each criterion. This convention differs from the earlier first-query benchmark, which cites the source with the largest positive margin.

- **Full batch:** Verify all comparisons in consecutive chunks of up to 16 pairs.
- **Two-wave full:** Verify source rank one for every candidate, then verify all criteria against source rank two.
- **Gap only:** Keep the first wave identical to two-wave full. Score second-source pairs only when the first wave did not establish that criterion for its owner.

Under stable pair verdicts, the first positive citation and matching rank are invariant when pruning second-source pairs for criteria already supported. A 1,400-case randomized local test checked that invariant, including missing sources, ties at the decision threshold, and single- and multi-criterion queries. Finite-precision batch geometry can still change live pair scores.

## RTX 5090 results

| Route | Scored pairs | Model calls | Model-supported experts | Changed ranking or citations against two-wave full | Verification P95 | Factored complete-route P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full batch | 1,280 | 115 | 70 | 3 / 80 | 172.4 ms | 2,108.9 ms |
| Two-wave full | 1,280 | 174 | 68 | 0 / 80 | 176.2 ms | 2,112.4 ms |
| Gap only | 1,180 | 173 | 68 | 0 / 80 | 147.4 ms | 2,081.3 ms |

Every arm performs its own live verification, with alternating execution order. The factored complete-route time adds its measured verification time to the common prefix measured once per query. The prefix includes parsing, query encoding and retrieval. This estimate excludes networking, service load and cold initialization.

Gap-only verification saves 100 pair scores and cuts verification P95 by 16.4% against two-wave full. Its factored complete-route P95 is 1.5% lower. Full batch adds two model-supported recommendations for different requests and changes ranking or citations on three requests relative to two-wave full, despite scoring the same pairs. This exposes batch-geometry sensitivity on the actual model. No pairs were truncated. The outputs were compared with a frozen verifier rather than human judgments.

Replay two second-source orders against the same two-wave scores under budgets of six and eight source reads. The fixed policy follows candidate rank after the first wave, and the adaptive policy prioritizes candidates with the fewest missing criteria. The two budgets preserve 59 versus 58 and 68 versus 66 reference recommendations across all 80 requests. Both paired bootstrap intervals include zero. In the 25 requests with multiple criteria, all policies retain the same three reference recommendations under either budget. Candidate retrieval, evidence availability and verifier errors require separate examination before drawing a quality conclusion about the scheduler. [Full aggregates](results.json)

## Reproduce

Run on a Slurm GPU compute node with the same development projections as the linked benchmark:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_gap_verification \
  --queries DEVELOPMENT_QUERIES --corpus DEVELOPMENT_SOURCES \
  --models CHECKPOINT_DIRECTORY --output NEW_RESULTS
```

Job 41141 completed with no sealed requests. The release contains aggregates and code.
