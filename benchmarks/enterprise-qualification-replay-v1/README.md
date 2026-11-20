# Isolate how eligibility interpretation changes enterprise abstention

Reuse the 25 multi-condition development requests from the
[first-round experiment](../enterprise-joint-first-round-v1)
and the exact predictions in the
[scope audit](../enterprise-qualification-scope-audit-v1). Verify
the original request, source and model hashes. Fix the five people
and each person's original two source prefixes. Keep the original
topic criterion in every path.

Score the original conditions in their original forward-batch layout
with the same 0.6B verifier. Separately score any newly introduced
qualification on those same excerpts. Reuse every margin across the
diagnostic arms. A candidate passes when every active condition has a
positive margin on at least one of their inspected excerpts.

| Arm | Eligibility conditions applied in addition to the topic |
| --- | --- |
| Original | All qualifications marked by the original parser |
| In both | Only qualifications accepted by both parsers |
| Direct gate | All qualifications accepted by the binary scope gate in the sampled excerpts |
| Union | All qualifications accepted by either parser |
| Topic only | None |

A separate calculation omits one original qualification at a time,
restricted to clauses the direct gate did not accept. All other original
qualifications remain. Report recoverable requests and unique
person–request combinations under that intervention.

This experiment changes interpretation only. The direct gate saw all
original qualifications and two extra sampled clauses per query.
Accordingly, its set cannot stand in for an exhaustive alternative
parser. Any newly issued candidate is still provisional and requires
human adjudication of the underlying eligibility interpretation.
The test does not re-run retrieval, measure latency or claim that
dropping an original condition is correct.

## Results

Both tasks of Slurm array `44885` on an RTX 5090 in `ws3` produce
byte-identical aggregates. The replay scores 740 distinct
condition–excerpt pairs.

| Interpretation | Requests with provisional results / 25 | Provisional people | Requests with no retained qualification |
| --- | ---: | ---: | ---: |
| Original | 0 | 0 | 0 |
| In both | 7 | 17 | 20 |
| Direct gate | 7 | 17 | 19 |
| Union | 0 | 0 | 0 |
| Topic only | 12 | 26 | 25 |

Omitting one disputed clause suffices to recover a provisional
candidate on four requests, involving 13 person–request
combinations. Every direct-gate result fails at least one original
model qualification check. The added clause accepted only by the
direct gate does not change the aggregate issuance counts from the
intersection path. Modeling the qualification set separately from
retrieval therefore explains part of the abstention observed in
this fixed run. It does not identify which original qualifications
were valid. Those questions remain assigned to the previously
frozen source-text review.

[Verified aggregate results](results.json).

## Reproduce

Use the raw aggregates and private traces from the two pinned GPU
benchmarks. Keep the frozen corpus, requests and trace outside Git.
Run on a Slurm GPU with a fresh output path.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_qualification_replay \
  DEVELOPMENT_QUERIES PRIVATE_CORPUS FIRST_ROUND_TRACE FIRST_ROUND_AGGREGATE \
  PRIVATE_SCOPE_TRACE SCOPE_AGGREGATE QWEN_RERANKER NEW_AGGREGATE
```
