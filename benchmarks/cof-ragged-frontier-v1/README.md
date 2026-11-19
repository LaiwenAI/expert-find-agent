# Unpadded evidence verification in the CoF frontier

Compare the ordinary padded four- and eight-row frontiers with eight-row verification that concatenates real tokens and uses separate causal attention segments. All routes have the same retrieved top five, two-source rule, two-token score projection, and original four-row full-head boundary rechecks. The packed implementation restores the ordinary attention backend before any recheck.

Packed sequences reset positions and use cumulative offsets at every Qwen3 attention layer. No pad tokens enter the dynamic model computation. Each prompt writes one yes/no margin from its own final token. The setup checks that altering the first independent prompt leaves the second prompt's margin unchanged. The live benchmark compares exact ordered experts and citations because different kernels can still differ numerically. The prototype switches a shared model setting during each dynamic forward and therefore requires sequential execution. A serving implementation needs an immutable backend per worker or a separate packed model path. [PyTorch variable-length attention](https://docs.pytorch.org/docs/stable/nn.attention.varlen.html)

The two public panels are reused. Component timing includes prompt construction, scheduler, dynamic verification and fixed rechecks. It excludes retrieval, startup, service transport and queueing. Token counts include padding during any reference recheck. Reviewer labels describe compatibility and carry no independent judgments for the selected citations.

## Results

Slurm job 42796 ran both panels on one RTX 5090 (PyTorch 2.15.0+cu132 and Transformers 5.23.0). All arms reproduce the complete ordered experts and citations on 122 KDD and 59 SciRepEval requests, with 12 and 21 labeled-positive cited recommendations.

| Panel | Route | Mean forwards | Mean processed tokens | Verifier P95 |
| --- | --- | ---: | ---: | ---: |
| KDD | Padded four | 3.26 | 6,692 | 124.50 ms |
| KDD | Padded eight | 2.28 | 6,930 | 128.58 ms |
| KDD | Unpadded eight | 2.24 | 5,900 | 104.77 ms |
| SciRepEval | Padded four | 3.83 | 6,550 | 111.74 ms |
| SciRepEval | Padded eight | 2.78 | 6,893 | 120.18 ms |
| SciRepEval | Unpadded eight | 2.80 | 5,931 | 92.56 ms |

Compared with the padded four-row baseline, unpadded execution lowers P95 by 15.8% and 17.2%. Mean latency savings per request have paired 95% intervals of 19.32–22.84 ms and 16.03–21.56 ms. Against padded eight, unpadded eight improves P95 by 18.5% and 23.0%, with similar counts of actual GPU forwards. This demonstrates a sequential verifier-component speedup with observed output parity. The comparison changes the backend as well as padding, so timing combines both effects. A concurrent service test and independent judgment of citation correctness remain separate.

[KDD aggregates](kdd.json) · [SciRepEval aggregates](scirepeval.json) · [Ordinary length-partition comparison](../cof-length-partition-v1)

## Reproduce

Use the same verified public inputs and checkpoints as the [wave-width experiment](../cof-frontier-wave-v1), in a Slurm CUDA allocation with PyTorch variable-length attention support.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.cof_ragged_frontier \
  KDD EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_KDD_RESULTS
PYTHONPATH=src:. python -m experiments.retrieval.cof_ragged_frontier \
  SciRepEval EMBEDDING_CHECKPOINT RERANKER_CHECKPOINT NEW_SCIREPEVAL_RESULTS
```
