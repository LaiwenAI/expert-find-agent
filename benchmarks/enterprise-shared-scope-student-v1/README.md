# Share scope encoding across qualification clauses

The [per-clause student](../enterprise-qualification-student-v1)
encoded a separate input for every clause. Most enterprise requests
still needed the 4B parser, and adding the student raised total
latency. This pilot asks how much cost is avoidable by sharing
the small model's request encoding. It also tests whether the
shared features retain scope cues on either side of the clause.

Fit three binary ridge heads on the same 96 bilingual
[scope-transfer examples](../enterprise-scope-transfer-v1).
Keep the open Qwen3-Embedding-0.6B backbone frozen.

| Encoding | Request representation at inference |
| --- | --- |
| Per clause | For each clause, encode the whole request as before, target and after fields, with the usual 512-token limit. |
| Shared forward | Encode the original clause order once. Use the hidden state at each clause end and the final document embedding. |
| Shared symmetric | Encode both clause orders together. Align by original offsets and average the normalized anchor and document features across the two orders. |

The shared routes normalize inter-clause whitespace to newlines
and keep the original text of each clause. The reversed route
reorders clauses without reversing tokens within clauses. The
right-context mechanism is related to discriminative token
classification in causal models. [Just Pass Twice (ACL 2026)](https://aclanthology.org/2026.acl-long.526/)
instead duplicates the input into a single sequence, so tokens
in the second copy can attend to the first. These experiments
compare the three routes above.

Set each confidence cutoff at the highest absolute score of a
classification error on the 24 calibration controls. Report
the 48 held scope-transfer views and the 72 different
[heading controls](../enterprise-heading-roles-v1) separately.
The public labels come from the authored contrasts. On
the 80 original private development requests, time all
clauses, including tokenization. A request would skip the
existing 4B consensus parser only if every clause is above
the frozen confidence cutoff and the full inputs fit. Compare
both whole-request acceptance and exact qualification sets.
The recorded time of a gated route adds the separately timed
4B parser for requests requiring fallback. It is a serial parser
simulation, without expert ranking, verification or client transport.
The enterprise comparison has model labels only. Report no
individual traces or restricted content.

## Results

Both Slurm tasks (array 47290, two RTX 4090s on ws6) reproduce
every decision count with reversed request and arm order.

| 0.6B encoding | Held prior qualifications found / 12 | Held false positives / 36 | Heading prior qualifications found / 36 | Heading false positives / 36 |
| --- | ---: | ---: | ---: | ---: |
| Per clause | 7 | 0 | 5 | 0 |
| Shared forward | 12 | 4 | 32 | 10 |
| Shared symmetric | 12 | 9 | 31 | 11 |

At the confidence cutoff, the per-clause head accepts no
targets in either held panel. The forward head accepts no
targets on scope transfer and four on heading controls.
The symmetric head accepts four and three, respectively.
All of these accepted targets are correctly classified.
Their small number limits what can be learned about risk.
Raw classifications and accepted classifications use distinct
denominators.

| Route | Student alone, P95 (ms) | With 4B fallback, parser P95 (ms) | Requests eligible to skip 4B / 80 | Changed qualification sets among skips |
| --- | ---: | ---: | ---: | ---: |
| Whole-block 4B consensus | — | 1,291–1,294 | — | — |
| Per clause | 462–466 | 1,737–1,741 | 7 | 1 |
| Shared forward | 22–23 | 1,305–1,309 | 5 | 0 |
| Shared symmetric | 50–51 | 1,314–1,318 | 16 | 2 |

Measured alone, the forward student's P95 is about 20 times
lower than the per-clause student's. Shared forward encoding
consumes 48,453 non-padding input tokens across all requests.
The per-clause prompts demand 1,428,015 tokens before clipping,
so their nominal count cannot be used as a processed-token
speedup estimate. All arms saw the same 1,452 target clauses.
The 4B parser marks 51 qualifying clauses. Before confidence
filtering, both the per-clause and forward heads make zero
positive marks on this enterprise cohort. The symmetric head
keeps two of the 4B marks and adds six other marks. These are
disagreements with a model reference. Each shared route still
increases parser P95 after its fallback cost. The experiment
supports shared encoding as a lower-cost feature extractor.
It provides no evidence that the current student can replace
eligibility decisions in the full service.

[Aggregate outputs and execution digests](results.json)

## Reproduce

Restore the pinned development projection from the
[hybrid benchmark](../enterprise-hybrid-rescue-v1) and the
Qwen3-Embedding-0.6B and Qwen3-4B-Instruct-2507 checkpoints.
Repeat on Slurm CUDA with `--reverse` and a fresh output path.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_shared_scope_student \
  DEVELOPMENT_QUERIES benchmarks/enterprise-heading-roles-v1/controls.json \
  benchmarks/enterprise-scope-transfer-v1/controls.json \
  QWEN3_EMBEDDING_CHECKPOINT QWEN3_INSTRUCT_CHECKPOINT NEW_RESULTS.json
```
