# Recheck contested clauses instead of an entire parser block

Keep the existing BF16/FP16 qualification-consensus trigger,
within-request prefix broadcast, live retrieval and evidence
verification. Change only the independent BF16 recomputation
after the trigger.

| Route | Clauses receiving independent BF16 scoring |
| --- | --- |
| Whole block | Every clause in a triggered block |
| Narrow | The two cached precisions disagree on mandatory qualification membership or the FP16 qualification gap is at most 0.5 |
| Wide | Narrow set plus clauses with BF16 qualification gap at most 2.0 |

Use the original BF16 cached output for unselected clauses.
Independent passes for selected rows see the same full request
context. The fallback forwards use fewer batch rows and can
therefore produce different floating-point scores. Reuse the
[existing 72 labeled synthetic controls](../enterprise-heading-roles-v1)
and 80 restricted enterprise development requests. Compare all
categorical parser labels, mandatory qualification offsets, and
the complete issued recommendation, citation and rescue outputs.
The live routes share the same automatic qualification scheduler
for source verification.

Full serial P95 includes parsing, embedding, retrieval, primary
verification and role-index fallback. Alternate arm order and
use the same resident models. Record original eligible rows,
recomputed rows and padded fallback token slots. The paired
interval resamples requests. Startup, serving queues, and
independent judgments of enterprise suitability are outside the
measurements.

## Results

Slurm job 43387 completed on one RTX 5090 on ws5. All
routes trigger seven independent rechecks. Both selective
rules choose the same seven rows on this replay.

| Route | Independent rows / eligible | Padded fallback token slots | Parser P95 | Complete P95 | Issued requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole block | 56 / 56 | 74,680 | 937.5 ms | 994.9 ms | 37 |
| Narrow | 7 / 56 | 9,250 | 602.9 ms | 711.0 ms | 36 |
| Wide | 7 / 56 | 9,250 | 602.9 ms | 710.7 ms | 36 |

Wide selection reduces fallback token slots by 87.6% and
full P95 by about 284 ms on this replay. The descriptive paired
P95 difference interval is −851.1 to +0.9 ms. Its median
complete latency is effectively unchanged. The timing improvement
is concentrated in requests that incur the expensive fallback.

The two selective routes change mandatory qualifications on
four development requests, and the role-index fallback output
on one. Primary issued responses and retrieved order stay
unchanged. The changed rescue output removes one issued
request. All arms label 60/72 anchored synthetic controls
correctly, with the same two false qualifications and ten missed
qualifications. Public controls alone would miss the production
shape differences observed on development. The positive
efficiency signal does not justify replacing block recomputation
with either mask in an output-preserving route. Human review of
the qualification differences remains needed. [Aggregate results](results.json)

## Reproduce

Use the restricted development data and pinned models from the
[enterprise qualification benchmark](../enterprise-shared-qualification-v1).
Run inside Slurm CUDA.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.retrieval.enterprise_clause_recheck \
  DEVELOPMENT_QUERIES DEVELOPMENT_CORPUS ROLE_WINDOWS VECTORS.npy \
  VECTOR_METADATA MODELS benchmarks/enterprise-heading-roles-v1/controls.json NEW_RESULTS
```
