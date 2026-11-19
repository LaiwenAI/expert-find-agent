# Length partitioning in frontier evidence verification

The [wave-width experiment](../cof-frontier-wave-v1) found that reducing the number of GPU calls could increase padding and tail latency. This experiment keeps the same ordered evidence frontier and changes how its requested work is packed on the GPU. Use the same full-inventory candidate generator, top five fixed candidates, two-source rule, two-token CUDA projection, and guarded full-head rechecks.

| Route | Dynamic verification |
| --- | --- |
| Frontier four | Original frontier, at most four documents per wave |
| Frontier eight | At most eight documents per wave, padded together |
| Bucket eight | Same eight-document frontier, sorted by prompt length and split into groups of at most four |
| Profiled eight | Same eight-document frontier, length sorted, then partitioned by dynamic programming to minimize the sum of measured shape costs |

Profile costs are medians over three real GPU forwards per shape, using repeated-token synthetic prompts and no CoF labels. Each shape is a batch of one to eight rows and a length from 64 to 1,024 in 64-token steps. For partition decisions, round each subgroup's maximum length up to the next profiled length. Actual inference uses dynamic padding as before. The profiler runs once before timing each dataset. Its cost is reported separately.

The three eight-document routes reveal the same set of scores at each frontier step when the numerical decisions coincide. Partitioning can change floating-point reduction order, so compare complete ordered experts and citations, as well as labeled-positive experts with model citations. The shared frontier continues to verify only sources needed by the two-document rule. The GPU reader's original fixed-shape four-row rechecks are reused in all arms.

Warm verifier timing includes prompt construction, planning, GPU inference and rechecks, and excludes retrieval, serving, queueing and initial profiling. The two reused public panels supply reviewer compatibility labels rather than independent citation judgments.

## Results

Slurm job 42791 ran both panels on one RTX 5090. All routes reproduce the ordered experts and citations on every request and retain 12 KDD and 21 SciRepEval labeled-positive recommendations with a citation.

| Panel | Route | Mean forwards | Mean processed tokens, with padding | Verifier P95 |
| --- | --- | ---: | ---: | ---: |
| KDD (122) | Frontier four | 3.26 | 6,692 | 125.45 ms |
| KDD | Frontier eight | 2.28 | 6,930 | 129.27 ms |
| KDD | Bucket eight | 3.26 | 6,581 | 129.40 ms |
| KDD | Profiled eight | 3.44 | 6,314 | 121.23 ms |
| SciRepEval (59) | Frontier four | 3.83 | 6,550 | 113.52 ms |
| SciRepEval | Frontier eight | 2.78 | 6,893 | 119.96 ms |
| SciRepEval | Bucket eight | 3.83 | 6,380 | 114.00 ms |
| SciRepEval | Profiled eight | 3.86 | 6,334 | 114.44 ms |

Profiling takes about nine seconds per panel. Compared with the four-row frontier, the profiled route saves mean latency in both panels (paired 95% intervals 2.67–5.94 ms and 1.60–6.21 ms), while tail improvement appears only on KDD. The outcome motivates measuring unpadded execution as a further route. These numbers compare sequential verification on the same device and do not establish a queueing or service benefit.

[KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json)

## Reproduce

Prepare the pinned public CoF data and frozen checkpoints described in the [wave-width experiment](../cof-frontier-wave-v1). Run inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_length_partition \
  KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_KDD_RESULTS
PYTHONPATH=src:. python -m experiments.retrieval.cof_length_partition \
  SciRepEval EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_SCIREPEVAL_RESULTS
```
