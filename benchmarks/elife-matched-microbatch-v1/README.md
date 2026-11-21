# Compare query and verifier microbatching in the expert-search service

Cross the [lightweight role cascade and calibrated source fallback](../elife-matched-source-service-v1)
with serial and opportunistic batched execution. Rebuild the same frozen
gates. Accept a timing comparison only when repeated queries and both
arrival rates reproduce the archived proof signatures.

The service worker takes the oldest HTTP request and drains up to
three already-queued requests in FIFO order. There is no fill timer.
Test batching at two stages independently.

| Stage | Work shared across queued requests | Work held per request |
| --- | --- | --- |
| Query embedding | Encode up to 16 topic and role prompts in one Qwen3 embedding forward | Ranking, planning, and verification |
| Paired role check | Batch deferred two-prompt checks in one 4B model forward | Embedding, ranking, source checks, and single-prompt role checks |

The role service sends each pair's own prompts to the verifier, splits
the result by request, and preserves that request's decision threshold
and evidence checks. Logical checks count against the existing
per-request budget. Physical verifier forwards count actual model
dispatches. A grouped request's service interval includes completion
of the group.

Use two Poisson rates, both arm orders, and eight presentations of each
request per arm and rate. Match the prior benchmark's arrivals on the
held and development panels. The client P95 includes queueing.
Quality is counted once per distinct query on the previously analyzed
structured panels.

## Results

### Query-embedding parity

The native-position arm (Slurm array 46347) and the request-specific
position arm (46353) both failed all four tasks on ws3 RTX 5090.
Each task observed a repeated query emit a different evidence response
within a replay. This reproduced on the held and development panels
and in both arm orders. The two variants did not pass the serial proof
invariant, so their latency comparisons are excluded. Preserving an
input's original token positions was insufficient. The experiment
does not isolate whether the remaining difference first arose in the
encoder, ranking, or the downstream decisions. See the
[Slurm records and failure reasons](embedding-parity-failures.json).

### Paired role verification

The held panel has 48 distinct queries, 12 paper and recorded-author
groups, and 384 requests per measured replay. The four policy and
execution combinations agree with their archived serial proof signatures.
The recovered keyword proof is retained. The cascade yields 10/12
strict keyword proofs and the calibrated fallback yields 11/12.
Both yield 12/12 strict excerpt proofs and zero observed adverse issuances.

| Arrival rate (req/s) | Policy | Serial P95 (ms) | Grouped P95 (ms) | Serial → grouped 4B forwards |
| ---: | --- | ---: | ---: | ---: |
| 31.03 | Cascade | 77.6–86.9 | 107.6–116.6 | 128 → 128 |
| 31.03 | Calibrated fallback | 82.7–88.2 | 100.2–104.5 | 144 → 144 |
| 66.50 | Cascade | 1334.3–1461.4 | 1419.6–1523.4 | 128 → 128 |
| 66.50 | Calibrated fallback | 1621.8–1754.3 | 1598.0–1680.0 | 144 → 140–141 |

The calibrated route saves 3–4 physical 4B forwards per held high-load
replay and lowers its P95 by 1.5%–4.2% within matched run orders.
At low load it saves no forward and P95 rises 13.6%–26.3%. The
cascade saves no physical forward at either rate and is slower with
group completion. One or no deferred check per queued group limits
the possible savings even when multiple HTTP requests are already queued.

The development panel has 108 distinct requests and 22 connected
paper and recorded-author groups. Each replay sends 864 requests.
The cascade and calibrated fallback independently match their archived
proofs and retain 12/27 and 14/27 strict keyword proofs, respectively.
All four Slurm tasks in array 46362 complete with zero request failures
across both panels.

| Arrival rate (req/s) | Policy | Serial P95 (ms) | Grouped P95 (ms) | Serial → grouped 4B forwards |
| ---: | --- | ---: | ---: | ---: |
| 29.79 | Cascade | 61.9–62.7 | 71.7–76.2 | 144 → 144 |
| 29.79 | Calibrated fallback | 63.1–66.7 | 78.9–79.4 | 168 → 168 |
| 63.84 | Cascade | 770.9–1173.2 | 936.4–1088.9 | 144 → 142 |
| 63.84 | Calibrated fallback | 1039.3–1363.6 | 988.3–1333.6 | 168 → 161–163 |

In the development panel, grouping reduces fallback P95 by 2.2%–4.9%
at high load and increases it by 18.3%–25.8% at low load. The cascade
has mixed high-load changes across matched orders. The grouped fallback
still has higher P95 than the serial cascade in both panels and both
orders. The small savings are conditional on saturated finite replay,
and throughput excludes the time before the trace. These results
support keeping the serial path as the default. A scheduler that admits
both single- and paired-role work and releases independent responses as
they finish would test whether the small dispatch savings can be increased
without the group-completion delay. See the [full runs](results.json).

## Reproduce

Use the [source fallback benchmark](../elife-matched-source-service-v1#reproduce)
with the same frozen inputs, one Slurm CUDA allocation per panel and
order, and a fresh output path. Add `--embedding-batch-capacity 4` or
`--role-batch-capacity 4` to that command in both orders. The embedding
command uses request-specific token positions. The initial pilot
used the same grouped execution with the encoder's default position IDs.
