# Fit a cheap qualification student from bilingual scope contrasts

Create synthetic paired scopes over twelve technical topics in
English and Chinese. Hold the target wording fixed while changing
who performed the work or whether it is past work or a future
deliverable. Include adjacent and intervening neutral context.
Use the first eight topics for fitting and the last four topics
for calibration. The same six layout cues appear across those topic
splits. These labels are generated from the construction.

Fit a ridge classifier on frozen Qwen3-Embedding-0.6B embeddings.
The target-only comparator receives the exact target clause.
The scope route also uses adjacent text and the nearest eligible
earlier heading. Only the ridge head is fitted on the GPU.
The binary output distinguishes mandatory prior qualification
from all other roles. Calibrate a symmetric confidence cutoff
as the largest absolute score among misclassified calibration
examples. The cutoff does not use the separately published
72 [heading controls](../enterprise-heading-roles-v1).

Use the scope head as a request-level guard on all 80 private
development requests. It skips the existing full parser only
when each clause exceeds the frozen confidence cutoff and no
compact input exceeds the embedding limit. Remaining requests
take the unchanged full parser. Both then run the same live
retriever, guarded automatic qualification verification, and
role-index fallback. Compare qualification spans, expert and
citation outputs, and realized pipeline timing. Raw head
agreement on private requests refers to the full parser, which
is a model reference. The private requests do not have human
clause labels. The sealed cohort is unused.

## Results

Slurm job `43414` completed on one RTX 5090. Encoding the
synthetic and published controls, fitting both ridge heads,
and scoring the calibration and control splits took 531 ms.

| Head | Calibration correct | Published controls correct | Published controls accepted after calibration | Errors among accepted controls |
| --- | ---: | ---: | ---: | ---: |
| Target only | 64/96 | 36/72 | 0/72 | 0/0 |
| Target plus scope | 88/96 | 67/72 | 52/72 | 0/52 |

These controls test whether the same clause can change roles
with different surrounding language. On private development
requests, the raw scope head disagreed with the 4B parser on
193 of 1,452 clauses: it missed 39 of the parser's 48 marked
qualifications and added 154. These are parser disagreements,
since there are no human labels on these clauses.

| Paired private development replay | Full parser | Student guard |
| --- | ---: | ---: |
| Requests skipping the full parser | 0/80 | 11/80 |
| Requests issuing at least one recommendation | 37/80 | 38/80 |
| Median complete latency | 175.6 ms | 196.7 ms |
| P95 complete latency | 991.1 ms | 1,153.1 ms |

The paired bootstrap 95% interval for the P95 change was
+33.7 to +167.4 ms. Two requests changed qualification spans
and primary recommendation payloads. Supplemental results
and initial vector-retrieval IDs were identical. The confidence
cutoff transferred to the published heading controls, but it
skipped the full parser too infrequently on the enterprise replay
to offset its added inference. Keep the full parser as the first-query
reference. Test whether a cheaper gate is justified only after
an independently judged sample can measure missed qualifications
and whether a gate preserves the issued recommendations.

## Reproduce

The runner generates the training and calibration contrasts from
its checked-in templates and records a content digest. Run on
an allocated CUDA node with the frozen embedding, 4B parser, and
reranker checkpoints used in the
[serial overlap comparison](../enterprise-first-query-overlap-v1).
The development queries, corpus, and indexed role evidence are
restricted. The aggregate report contains counts and timing only.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_qualification_student \
  "$DEVELOPMENT_QUERIES" "$SOURCE_CORPUS" "$ROLE_INDEX" \
  "$VECTORS_NPY" "$VECTOR_METADATA" "$MODEL_ROOT" \
  benchmarks/enterprise-heading-roles-v1/controls.json "$RESULT_JSON"
```
