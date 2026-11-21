# Classify qualification scope in one request encoding

The per-clause student repeats most of the request for each target.
The [shared-scope student](../enterprise-shared-scope-student-v1)
uses both the original and reversed clause order. Its
[mixed-scope follow-up](../enterprise-mixed-scope-student-v1)
never classifies all views of any held mixed-scope anchor correctly.
This study tests one full-context forward with a clause-level readout.

Retain original wording, inter-clause whitespace, and source offsets.
Attach the readout to the final token overlapping each clause.
The bidirectional mask allows each anchor to attend to every real
token in the request. Check this by changing text after a fixed
anchor and comparing its representation with a causal control.
Full-context attention on decoder backbones has prior work, including
[Bitune](https://aclanthology.org/2025.emnlp-main.481/) and
[Look Both Ways and No Sink](https://aclanthology.org/2025.acl-long.1113/).
[Just Pass Twice](https://aclanthology.org/2026.acl-long.526/) gives
a causal model full right context by concatenating the input with itself.

| Arm | Attention | Trainable weights |
| --- | --- | --- |
| Bidirectional head | Full request | Binary head |
| Causal LoRA | Left context | Same head and rank-eight Q/V adapters in the final three layers |
| Bidirectional LoRA | Full request | Same head and adapters |

Use Qwen3-Embedding-0.6B and the same 576 labeled anchors from
192 constructed mixed-scope requests, six epochs, one training
seed, and the same weighted binary loss. Fit to authored labels only.
The 4B parser's outputs on private requests are a development
reference for comparison. Preserve template and topic separation
across synthetic training, calibration, and held splits. Measure
qualification recall, false marks, exact sets on annotated anchors, and
consistency across the topic-preserving scope changes. Report the
separate 72 heading controls.

Fix the confidence cutoff on the 144 mixed calibration anchors as
the maximum absolute score of any error. In the enterprise replay,
skip the 4B parser only if all clause decisions clear that cutoff.
Report skip-count, disagreement among skips, classifier latency,
and estimated serial parse time with fallback. Parser-equivalent
output is a model disagreement measure. There are no independent
enterprise clause judgments. The timing excludes retrieval,
evidence verification, and request transport.

## Results

Both RTX 4090 tasks in Slurm array 47430 reproduce the losses and
decisions in opposite replay orders. Rewriting the text after a
fixed anchor leaves the causal representation unchanged and changes
the full-context normalized representation by 0.140 in the largest
coordinate. The full-context intervention is active.

| Arm | Held mixed qualifications found / 72 | Held mixed false marks / 216 | All views correct / 36 scope contrasts | Both views correct / 36 heading pairs |
| --- | ---: | ---: | ---: | ---: |
| Bidirectional head | 21 | 20 | 3 | 1 |
| Causal LoRA | 27 | 50 | 4 | 11 |
| Bidirectional LoRA | 15 | 26 | 0 | 2 |

The roughly 124,000 trainable parameters in each LoRA arm reduce
weighted training loss below 0.004. The held scope comparisons give
little evidence of generalization. With the frozen cutoff,
bidirectional LoRA makes 16 errors among 63 accepted held mixed
anchors and 14 among 28 accepted heading controls. The other arms
accept none on either panel. High calibration confidence is
insufficient to trust a qualification decision across layouts.

| Arm | Development marks also found by 4B / 51 | Additional marks absent from 4B | Requests skipping 4B / 80 | Student P95, ms | Parser P95 with fallback, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen 4B parser | 51 | 0 | 0 | — | 1,290–1,294 |
| Bidirectional head | 0 | 0 | 0 | 28–30 | 1,309–1,312 |
| Causal LoRA | 2 | 8 | 0 | 24–25 | 1,307–1,310 |
| Bidirectional LoRA | 0 | 0 | 0 | 28–30 | 1,309–1,313 |

The one-pass student is inexpensive on this hardware, and its
fallback gate adds time to every request. Retain the existing
whole-block qualification parser. This result motivates supervision
with representative scope layouts and independent span judgments
before further threshold or model changes. [Aggregate runs](results.json)

## Reproduce

Use the frozen development projection and 4B reader from the
[hybrid benchmark](../enterprise-hybrid-rescue-v1) and the open
embedding checkpoint from the shared-scope pilot. Run in a CUDA
Slurm allocation. Repeat with `--reverse` for reversed replay and
arm order, using the same training seed.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_bidirectional_scope \
  DEVELOPMENT_QUERIES benchmarks/enterprise-heading-roles-v1/controls.json \
  benchmarks/enterprise-scope-transfer-v1/controls.json \
  QWEN3_EMBEDDING_CHECKPOINT QWEN3_INSTRUCT_CHECKPOINT NEW_RESULTS.json
```
