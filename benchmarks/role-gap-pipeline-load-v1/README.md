# Overlap CPU evidence preparation with a canonical GPU lane

The serial [prefix reuse](../role-gap-prefix-reuse-v1) and
[selective recheck](../role-gap-selective-recheck-v1) paths changed
some ordered witnesses. This experiment leaves the canonical input
shapes and scoring arithmetic in place. Overlap CPU source selection
and reranker tokenization with GPU work for another request.

Use the 110 eLife requests from the [four-source replay](../role-gap-title-refill-v1)
at full available declaration visibility. Use one warm GPU lane in
both arms. In the serial arm, the lane finishes a complete request
before starting another. In the pipelined arm, query encoding sends
the frozen vector to one CPU preparation worker. When the candidates
and tokens are ready, verification takes priority over admitting a
new request to the encoder. The GPU runs one operation at a time.
Each verifier call retains the original four sources, token order,
padding and weights. No cross-request model batching or persistent
query-result cache is used.

Pair the two implementations on an identical independent Poisson
schedule within each seed. For each of two independent seeds, visit
all 110 queries twice in independently permuted order. Swap the arm
order between seeds. Use nominal offered rates of 18 and 36
requests per second and one small unmeasured warm-up per arm.
Latency begins at the scheduled arrival and ends at serialized
witnesses. Report arrival lag, final drain and completed rates
alongside P95. The program checks aggregate frozen serial witnesses
and compares ordered response bytes for every paired arrival.
Timing is single-process and includes queueing, query encoding,
retrieval, tokenization, verification and witness serialization.
There is no network, parser, remote I/O, or production traffic.

## Results

Jobs `44074` and `44075` on ws5 reproduce all 1,760 paired
responses byte-for-byte. Each row pairs the same 220 scheduled
requests. Times include the finite trace and queue drain.

| Observed offered requests/s | Serial P95, first / repeat | Pipelined P95, first / repeat |
| --- | ---: | ---: |
| 16.07 | 58.58 / 58.61 ms | 57.96 / 58.01 ms |
| 17.33 | 85.88 / 85.23 ms | 83.45 / 82.89 ms |
| 32.14 | 191.67 / 192.23 ms | 185.06 / 187.17 ms |
| 34.66 | 308.15 / 308.03 ms | 280.40 / 265.83 ms |

The denser traces have 2.6% to 13.7% lower P95. The lighter ones
have 1.0% to 2.8% lower P95. This is a queueing result: the
number of model calls, verifier inputs, and ordered recommendations
are identical, and host time spent in encoder/verifier calls does
not fall. Timing excludes cold start and transport.
The completed rates and recorded drain characterize these traces,
without identifying sustained service capacity.

Use the overlapped schedule as an output-preserving execution
comparison for this replay, including when measuring retrieval
policy differences. This measures one public role-search pipeline;
an enterprise request also parses qualifications and can invoke
further evidence retrieval, which this replay excludes.

[Aggregate results](results.json)

## Reproduce

Use the pinned eLife catalog and MEDLINE assignment snapshot from
the gap-refill study. Run the benchmark on one Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.role_gap_pipeline_load ELIFE_PROJECTION \
  benchmarks/elife-mesh-v1/mesh.json benchmarks/role-gap-title-refill-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
