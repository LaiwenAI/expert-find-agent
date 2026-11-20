# Benchmark live contributor search over loopback HTTP

Serve the public PLOS/eLife joint request path from the
[in-process benchmark](../joint-request-queue-v1). The driver
sends the same 211 distinct requests in four separately shuffled
repetitions over persistent HTTP/1.1 on 64 loopback connections.
Each request includes its publisher catalog and the two topic–
role requirements. The handler parses the request and submits
it to one GPU worker. The worker runs live encoding, hybrid
ranking, cross-publication nomination, three warm declaration
file reads and JSON response serialization. Clients read full
responses and compare their exact bytes.

Compare one request per forward, dynamically batch up to eight
requests under a 2 ms deadline from the oldest request's arrival
at the server, and use the same scheduler with padding to a
fixed 16-topic batch. All topic inputs are 64 tokens wide
with positions relative to the attention mask. Match each
successful response against two independently computed references.
The serial reference uses one two-topic request. The padded
reference gives the same request 14 additional duplicate topic
rows, with those outputs discarded.

Replay seeded Poisson arrivals at nominal 60, 150 and 280
requests/s and report the realized trace rates. The same
scheduled arrivals and query order apply across policies.
Client P95 spans scheduled arrival to the last response byte.
The HTTP round-trip timer starts when a client connection
actually sends the body. Report queueing at the client and
server separately. Each client stops waiting after eight seconds from the
scheduled arrival. Count deadline misses
and other failures independently from successful latency
percentiles. Achieved rate divides successful completions by
time to drain the entire trace. Startup and offline index or
article encoding are outside the timer.

The server binds to loopback and uses the repository's existing
HTTP transport utilities. The experiment includes real TCP,
HTTP parsing, persistent clients and a worker queue. External
networking, TLS, multiple workers, remote storage and concurrent
business traffic are outside the measurement. Structured query
requirements and publisher declarations are shared across all arms.

## Results

Both Slurm tasks in array `44520` completed on ws5 using one
RTX 5090. Each cell reports the range across two execution
orders. All 844 requests per arm completed without deadlines
or other errors in every trace.

| Realized arrivals (requests/s) | Serial P95 (ms) | Dynamic batch 8 P95 (ms) | Padded 8 P95 (ms) |
| ---: | ---: | ---: | ---: |
| 58.49 | 37.59–37.68 | 20.18–20.30 | 25.12–25.43 |
| 146.21 | 2476.89–2569.08 | 21.65–22.18 | 26.64–26.68 |
| 272.93 | 5061.90–5189.91 | 23.49–23.57 | 27.68–27.94 |

At 272.93 arrivals/s, serial drains at 99.02–100.59 requests/s
and dynamic batch 8 drains at 271.14–271.26. Client-side queue
P95 exceeds four seconds for overloaded serial dispatch, where
the 64 connections also limit the number of outstanding requests.
The result measures backlog reduction when crossing server
capacity in a short open-loop trace. The low load illustrates
behavior when all policies handle the full incoming rate.

At the high load dynamic batching changes as many as five
of 844 responses versus the serial reference per repeat. It
changes one per repeat even at the 58.49 requests/s rate.
Padding reproduces the 16-row reference byte-for-byte in
every arm and load that uses fixed 16-row execution. The
16-row and 2-row references differ for one distinct request.
That query occurs four times in each trace, so all padded
traces differ from the serial reference on four responses.
Production equivalence requires a defined reference and
longer tests with real workloads.

[Complete latency, failure and parity results](results.json).
The `network_ms` field measures full HTTP round-trip time from
client send, including parsing, queues and server computation.

## Reproduce

Restore the public publisher snapshots and the pinned Qwen3
embedding. Run within a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.joint_loopback_http \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING NEW_RESULTS
```

Repeat with `--order-offset 1`.
