# Audit whether enterprise source fields support declaration-based pruning

The [PLOS replay](../plos-stable-topic-compaction-v1) skips
sources that are known to lack the requested same-person roles.
Measure how often the current enterprise development search
actually reaches structured or explicit personal-role evidence.

Join every pinned live source with its exact owner and text in
the private development export. Verify the original manifest.
Inspect the original value types and apply the existing
[project and patent marker parser](../enterprise-project-role-rescue-v1).
A delimited project-clause marker is a profile claim. Count exact
normalized owner names occurring in such clauses separately.
The inventor indicator requires an exact normalized owner match
within an inventor list. These indicators track what the profile
says. They carry no roster-completeness or independent confirmation
judgment. The scanner is deliberately narrow and can miss implied
roles, aliases, and nonstandard formats.

Replay the complete automatic enterprise route on the pinned 80
development requests in Slurm CUDA. Record the source ID and
offsets in the actual primary verification plan. Count unique
excerpts within each request and report criterion–excerpt pairs
separately. Count emitted primary and recovery citations by the
same indicators. No variant suppresses any candidate; the
counterfactual counts how many baseline recommendations contain
an excerpt with no detected role cue. The cutoff-specific regex
on parsed prior qualifications is a cue diagnostic.

Only aggregate counts are published. Source IDs, exact spans,
names, and enterprise request text stay in the authorized workspace
and never appear in the public result. The independent human
review cohort stays sealed.

## Results

Job `43483` ran on one RTX 5090 with the pinned current
search stack. The 52,667 source inventory has 1,832 project fields
and 713 patent fields, all originally strings. Across the full
inventory, 1,093 records have a detected role cue. Twenty-one
project records have an exact owner name in a role-bearing clause,
and nine patent records have an exact owner in an inventor list.

| Stage | Denominator | Any detected role cue | Exact-named inventor |
| --- | ---: | ---: | ---: |
| First five candidates, looking anywhere in their profile | 400 slots | 130 | 3 |
| Actual primary excerpts | 800 excerpts | 70 | 1 |
| Planned primary verification | 1,280 criterion–excerpt pairs | 115 | 2 |
| Issued provisional citations | 73 citations | 8 | 0 |

The published counts for candidate slots sum over requests.
Exact-name project clauses cover one candidate slot anywhere
in the profile, zero primary excerpts, and zero issued citations.
Of 48 parsed prior qualifications, none matches the fixed duty
cue regex. This is a pattern count on parser-selected clauses.
Baseline search issues 73 provisional recommendations on 37
requests, and 65 recommendations have only citations without
a detected cue.

The current records support using detected roles for positive
evidence retrieval and source enrichment. Dropping excerpts when
a role cue is absent would remove all current witnesses from
65 model-issued recommendations. Their relevance and ownership
still need independent adjudication. The structured PLOS pruning
criterion requires a completeness signal separate from cue detection.
[Aggregate results](results.json)

## Reproduce

The first command runs locally against the authorized private
development snapshots. The second runs in a Slurm CUDA allocation
with the resident vector bank and frozen open checkpoints from
the [live role-index benchmark](../enterprise-role-first-query-v1).

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.enterprise_declaration_reach prepare \
  PRIVATE_DATASET DEVELOPMENT_SOURCES PRIVATE_CUE_PROJECTION
python -m experiments.retrieval.enterprise_declaration_reach run \
  DEVELOPMENT_QUERIES DEVELOPMENT_SOURCES ROLE_WINDOWS PRIVATE_CUE_PROJECTION \
  VECTORS_NPY VECTOR_METADATA MODEL_ROOT NEW_AGGREGATES
```
