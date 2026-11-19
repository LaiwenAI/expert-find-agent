# Reduce serial waves in evidence verification

The previous [binary projection experiment](../cof-binary-head-v1) found that compressing the verifier's output head saves about a millisecond. This experiment tests launch scheduling with the same shared-frontier verifier, two-token head, and batched tokenization. It uses all 122 KDD and 59 SciRepEval requests from the same full-inventory CoF protocol.

A frozen full-inventory ranker supplies five experts and up to three candidate sources each. The existing shared frontier defers reading sources until their verdicts can affect the ordered result. It requires two positive documents for a cited recommendation. Guarded scores near zero still use the original fixed-shape four-row recheck. A source is scored once per request and shared across its documented owners.

Compare a four-row or eight-row limit for ordinary dynamic verification waves. For either limit, the optional fill policy reserves empty rows for unread sources from an expert already present in the wave. It considers experts that can still reach the two-source requirement. Added sources cannot increase the longest prompt length of the active batch. Their votes enter the decision only if the original frontier eventually requests the source. The scorer counts all actual work, including speculative reads that the stopping rule never uses.

The four-row baseline preserves the exact previous schedule. The arms share candidate retrieval and model parameters. Request order rotates. Report exact ordered experts and citations, published positives accompanied by model-selected citations, GPU forwards, documents, padded tokens and verifier P95. Timings include tokenization and all live verdict computations, excluding candidate retrieval, service transport and waiting. Released expert labels do not judge each citation.

## Results

Slurm job 42782 ran both public panels on one RTX 5090. Each arm reproduces all 181 ordered results and citations. The 122 KDD and 59 SciRepEval requests retain 12 and 21 cited positives respectively.

| Panel | Policy | Mean forwards | Mean scored sources | Mean padded tokens | Mean unused extra sources | Verifier P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KDD | Four rows | 3.26 | 9.70 | 6,692 | 0.00 | 125.37 ms |
| KDD | Four rows + fill | 3.12 | 9.90 | 6,846 | 0.20 | 126.34 ms |
| KDD | Eight rows | 2.28 | 9.63 | 6,930 | 0.00 | 128.85 ms |
| KDD | Eight rows + fill | 2.16 | 10.21 | 7,316 | 0.58 | 131.71 ms |
| SciRepEval | Four rows | 3.83 | 11.71 | 6,550 | 0.00 | 112.33 ms |
| SciRepEval | Four rows + fill | 3.69 | 11.97 | 6,774 | 0.25 | 113.54 ms |
| SciRepEval | Eight rows | 2.78 | 11.51 | 6,893 | 0.00 | 120.77 ms |
| SciRepEval | Eight rows + fill | 2.46 | 11.92 | 7,088 | 0.41 | 119.05 ms |

Widening without fill reduces launches by about 30% and 27% on the two panels and slightly reduces the number of documents scored, while processing 3.6% and 5.2% more padded tokens. This identifies padding as an observable countervailing cost. The paired mean latency differences for widening alone include zero on both panels. The four-row fill increases mean latency on both panels, with paired intervals of +0.51 to +1.48 ms and +0.32 to +2.92 ms. Additional shorter rows still incur processing cost even without increasing the batch's longest prompt. Keep the four-row policy as the sequential timing reference. Evaluate length-compatible packing or reducing backbone work next.

[KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json)

## Reproduce

Use the pinned checkpoints and public snapshots from the [binary projection experiment](../cof-binary-head-v1), on a Slurm CUDA compute node.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_frontier_wave \
  KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_KDD_RESULTS
PYTHONPATH=src:. python -m experiments.retrieval.cof_frontier_wave \
  SciRepEval EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_SCIREPEVAL_RESULTS
```
