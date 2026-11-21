# Measure role prompt latency through loopback HTTP

Replay the [393-article eLife holdout](../elife-prompt-holdout-v1)
with the same role labels and frozen model settings. Mix excerpt
and keyword views with both sources available. A client sends two
topic descriptions and two role phrases as JSON. The server
receives no labels, target article identifiers or contributor
keys. It returns its selected source identifiers, candidate
contributors and role sets. The offline client scorer checks the
actual response for the intended sources, exact contributor set,
and both declared roles.

Warm the resident models, index and request path. Use one FIFO
GPU worker for all methods, without request or response caching.
Compare the static prototype band, proof-gated full definitions,
and proof-gated role names. Keep the candidate bands, source
anchor, branch skip and independent role agreement the same
for both reader prompts. Each arm uses the same Poisson arrival
trace at nominal rates of 35, 65 and 80 requests per second.
Repeat the case mixture twice, rotate arm order between rates
and reverse the order in the second job. The load client uses
64 persistent local connections and includes time from scheduled
arrival. It records request deadline failures separately from
completed-request latency. Report achieved throughput over the
whole run, including queue drain.

## Results

All 528 requests complete per load and per arm in both Slurm tasks
of array 45965 on the ws3 RTX 5090. The measured offered rates differ
from the nominal rates because each replay uses one finite Poisson
trace. Quality counts are consistent across the rates and trials.

| Offered requests/s | Static client P95 | Full definition client P95 | Role name client P95 | Achieved requests/s, static / full / name |
| ---: | ---: | ---: | ---: | --- |
| 38.4 | 38.6–38.8 ms | 84.9–88.1 ms | 64.0–66.2 ms | 38.4 / 38.4 / 38.4 |
| 71.2 | 138.2–158.1 ms | 929.0–1051.4 ms | 385.3–484.6 ms | 70.9–71.0 / 62.0–62.9 / 66.6–67.9 |
| 87.7 | 504.1–628.2 ms | 2211.0–2353.3 ms | 1664.8–1767.4 ms | 78.9–80.5 / 62.1–63.2 / 67.0–67.9 |

The name prompt transmits 9,816 model input tokens across 102 live
reader calls, down from 24,600 for the full prompt. The server's
compute P95 is about 25–26 ms for names and 37 ms for full definitions.
At 71.2 arrivals/s the gap grows at the client as requests queue.
Both reader arms need to drain work after the last arrival. At
87.7 arrivals/s the static route also accumulates work. Measured
throughput includes the drain and is a finite replay statistic.

| Policy | Strict proofs / 132 distinct positive views | Adverse / 264 distinct requests | Adverse / issued |
| --- | ---: | ---: | ---: |
| Static prototype | 61 | 5 | 5/67 |
| Full definitions | 73 | 7 | 7/80 |
| Role names | 75 | 7 | 7/82 |

The role-name and full prompts produce different evidence responses
on two of 264 distinct requests. Both are strict proof gains. The
same-query outputs are stable across rates. The recovered proofs
match the corresponding held excerpt and keyword totals. No route
meets a 5% adverse share among issued requests on these descriptive
counts. The rates use the same limited 13 article pairs and carry no
population-level inference. This benchmark covers a single FIFO
worker with warm models, loopback connections, and pre-split query
conditions. It excludes remote network, cold start, and unstructured
request parsing. [Complete service measurements](results.json)

## Interpretation

Each replay repeats 264 distinct requests, including 132 positive
views. Repeating them estimates queueing under load and contributes
no additional independent proof labels. The client counts adverse
results both against all arrivals and against issued recommendations.
A shorter prompt may preserve aggregate error counts while changing
individual responses. The indexed corpus, request representation,
worker, offered arrivals and role agreement guard are shared.

## Reproduce

Restore the original eLife holdout projection using its manifest.
Use the same open checkpoints and frozen settings as the prompt
comparison. Run inside a Slurm CUDA allocation with a fresh output
path. The HTTP server binds only to the allocated node's loopback.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.elife_role_prompt_service \
  ELIFE_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json NISO_EXAMPLES \
  benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  benchmarks/engineering-one-role-recheck-v1/results.json \
  QWEN_EMBEDDING QWEN_4B NEW_RESULTS
```

Repeat with `--reverse`.
