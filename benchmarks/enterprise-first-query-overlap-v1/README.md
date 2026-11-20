# Overlap demand parsing and candidate retrieval within the first query

The enterprise retriever consumes the complete original request.
It does not use the parser's derived qualification offsets.
The verification stage needs both inputs. This comparison
starts retrieval on a worker thread and a separate CUDA stream,
then parses on the original stream. It joins both outputs before
running the unchanged automatic qualification schedule and
role-index fallback.

The concurrent path uses one RTX 5090 and one request at a time.
There is no cross-request cache, microbatching, or speculative
candidate verification. The ordered list of experts and sources,
including scores, is compared with the serial path along with
all parser labels, qualifications, evidence decisions, and issued
recommendations. Request processing order alternates between
arms. The data are the same 80 development requests and pinned
52,667-source inventory.

Measure warmed serial request latency. Stage windows include host
work, waits, and GPU operations. Measured overlap between two
stage windows is wall-clock overlap, not a GPU kernel utilization
measurement. Report the join time and total time together since
resource contention can slow both concurrent stages. Paired
intervals resample requests, and exclude service queues and
transport.

## Results

Slurm job 43401 completed on one RTX 5090 on ws5. All 80
requests reproduce the exact retrieval scores and order,
clause labels, mandatory qualification spans, expert and
citation outputs, and fallback behavior. Both arms issue
provisional recommendations on 37 requests.

| Route | Parser P50 / P95 | Retrieval P50 / P95 | Join P50 / P95 | Complete P50 / P95 |
| --- | ---: | ---: | ---: | ---: |
| Serial | 116.2 / 938.4 ms | 9.0 / 10.0 ms | 125.0 / 950.1 ms | 176.4 / 996.9 ms |
| Parallel | 121.1 / 947.4 ms | 25.0 / 38.2 ms | 121.2 / 947.4 ms | 172.8 / 995.6 ms |

The parallel arm overlaps the retrieval stage window with the
parser, but both stages take longer under concurrency. Its
complete P95 point estimate is 1.25 ms lower, with a
descriptive paired difference interval of −14.1 to +1.4 ms.
Parsing dominates the serial tail. Overlapping this already
short retrieval stage gives little observed tail benefit
on one GPU. This is a decision-equivalence and timing
experiment, with independent matching quality and sustained
throughput outside its scope. [Aggregate results](results.json)

## Reproduce

Use the authorized development inputs and models from the
[qualification benchmark](../enterprise-shared-qualification-v1).
Run on a Slurm CUDA node.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_first_query_overlap \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS ROLE_WINDOWS VECTORS.npy \
  VECTOR_METADATA MODELS NEW_AGGREGATE_RESULTS
```
