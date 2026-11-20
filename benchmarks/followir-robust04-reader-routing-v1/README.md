# Route fixed rescue requests to the 4B short reader

The [reader factorial](../followir-robust04-reader-factorial-v1)
found more headroom with a 4B reader and a short window. Running
it for every rescue request increased local P95. Here the same
model, prompt, cutoff, title and 850-character body window define
the always-4B reference. Preserve the original first-pass gate
and the same pair of rescue proposals in every arm.

Compare three request-level gates. The lexical gate uses the
largest coverage of instruction terms not present in the topic
query across the two lead windows. The 0.6B gate uses the largest
full-criteria margin from its two-candidate batch. The combined
gate requires both. A skipped request returns no rescue document.
An invoked request sends the original pair to 4B, keeping the
same batch shape and acceptance threshold.

Pick the largest threshold for each feature that still invokes
4B for every calibration request on which the reference issues
a result. This preserves all reference decisions by construction
on Core17 and News21. The policy is frozen before replaying the
previously explored Robust04 workload. This calibration target
is exact reference retention, separate from document relevance
and any population risk claim.

Measure the live verification stage three times in rotated
order. P95 includes local window construction and every model
call actually made. It excludes first-pass retrieval, selection
of candidates, queueing, transport and model loading. Compare
accepted positive, negative, invalidated and unjudged documents,
whole-batch output agreement and absolute calls. Report
per-topic uncertainty for any lost known positives.

## Results

Slurm job `43663` ran on one RTX 5090. Both calibration
and test reference counts reproduce the factorial run.
Robust04 has 45 rescued requests out of 50 eligible topics.
Every invoked batch returns the original reference decision.

| Route | 4B calls / 45 | Exact outputs / 45 | Positive / negative | Mean | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always 4B | 45 | 45 | 4 / 0 | 45.12 ms | 54.62 ms |
| Lexical | 40 | 45 | 4 / 0 | 40.15 ms | 54.12 ms |
| 0.6B | 10 | 44 | 2 / 0 | 21.22 ms | 61.59 ms |
| Both | 10 | 44 | 2 / 0 | 19.90 ms | 61.42 ms |

The lexical cutoff is 1/21 instruction term coverage. It skips
five empty reference outputs and reduces mean verification
time by 11.0%. The measured P95 difference is 0.50 ms; retained
4B calls still dominate the tail. The 0.6B cutoff of 1.25 removes
35 calls, yet it drops two known positives from one request
and adds a second forward where 4B remains necessary. The
combined gate makes the same decisions and 4B calls here.
The per-topic loss is 0.04 positives for either 0.6B gate
(descriptive 95% interval 0.00 to 0.12).

All methods produce zero known negative and unjudged acceptances
at this reference threshold. The lexical gate is a same-output
mean-time control on a reused cohort. Four positive documents
are insufficient to infer broad retention, and the time
measurements do not establish end-to-end latency savings.
These results keep the 0.6B precheck as an efficiency contrast
rather than a default route.

[Aggregate results](results.json)

## Reproduce

Use the same pinned inputs from the
[Robust04 calibration study](../followir-robust04-rescue-risk-v1)
and the frozen model checkpoints from the factorial. Run in a
Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.followir.robust04_reader_routing \
    CORE17_JSON CORE17_VECTORS NEWS21_JSON NEWS21_VECTORS \
    ROBUST04_JSON ROBUST04_VECTORS QWEN_EMBEDDING QWEN_RERANKER QWEN4B \
    benchmarks/followir-constraint-rescue-scope-v1/results.json \
    benchmarks/followir-robust04-rescue-risk-v1/results.json \
    benchmarks/followir-robust04-reader-factorial-v1/results.json NEW_RESULTS.json
```
