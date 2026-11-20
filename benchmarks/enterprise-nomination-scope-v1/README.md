# Separate candidate nomination from qualification interpretation

Use the 25 multi-condition requests from the frozen
[development scope replay](../enterprise-qualification-replay-v1).
Compare candidates and exact excerpts from the
[condition-aware nomination experiment](../enterprise-condition-nomination-v1).
Keep the topic, qualification offsets, and 0.6B verifier fixed
for each input. The nomination policies keep their original
ranked people and exact excerpts throughout.

Score every topic and original qualification using the
reference batch order. Score additional clauses in a separate
batch when the sampled alternative marks them as qualifications.
Reuse these scores across interpretations. Every policy has the
same verification exposure. The alternative classifies all
original qualifications and up to two other sampled clauses
per request. It is neither a complete reinterpretation nor a
human qualification judgment.

## Results

Both Slurm tasks in array `45208` reproduce the same results
after reversing request order. Each nomination and excerpt
policy evaluates 740 criterion–excerpt pairs. The table reports
requests with at least one model-approved expert and the total
number of model-approved people in parentheses.

| Candidate nomination | Original scope | Sampled alternative | Union scope | Topic only |
| --- | ---: | ---: | ---: | ---: |
| Hybrid | 0 (0) | 7 (17) | 0 (0) | 12 (26) |
| Per-condition | 1 (1) | 10 (17) | 1 (1) | 15 (26) |

Hybrid prefixes and hybrid joint selection produce the same
counts. The same holds for both per-condition excerpt policies.
Under the sampled alternative, per-condition nomination yields
model approval on three more requests with no previously
approved request lost, while the total approved people stays
at 17. Sixteen of those 17 do not satisfy all checks under the
original interpretation. Under the union, require every
qualification selected by either interpretation. Only one
request has a model-approved candidate in that comparison.

Within this development panel, the difference between the
interpretations is larger than the gain from candidate
replacement at fixed read depth. Model agreement cannot decide
which clauses impose a hard qualification. Independent
review must resolve disputed language and establish whether an
exact evidence view supports the relevant person and criterion.
[Aggregate results](results.json).

## Reproduce

On a Slurm GPU, reuse the pinned development inputs and the
exact private traces from first-round search, condition-aware
nomination and the sampled qualification comparison.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_nomination_scope \
  QUERIES SOURCES BASE_AGGREGATE BASE_TRACE NOMINATION_AGGREGATE NOMINATION_TRACE \
  SCOPE_AGGREGATE SCOPE_TRACE VERIFIER_CHECKPOINT NEW_RESULT
```

Repeat with `--order-offset 1`.
