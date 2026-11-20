# Expand same-candidate evidence beyond role-tagged sources

The [role-only audit](../enterprise-gap-bound-recovery-v1)
found no provisional completion after scoring 58 available role
windows for partially supported candidates. Replay the frozen
enterprise primary search on the authorized 80 development
requests. Use the 14 requests whose primary step abstains while
at least one of its five candidates has evidence for some criteria.
Keep their qualification parsing, candidate shortlist, prior
positive citations and verifier fixed.

For each missing criterion and candidate, allow at most two new
criterion–excerpt checks. Compare lexical selection of existing
role windows, lexical selection across the candidate's whole
catalog, and dense source selection across that same catalog. For
the wider sources, expose up to 520 characters per article or
profile field, chosen by section boundaries and lexical overlap
with the missing criterion. The dense arm ranks sources by
their already indexed vector and a single batched encoding of
the distinct missing criteria. For open-scope queries, exclude
source IDs already read by the primary step. An explicit
same-object cue restricts new snippets to the source already
supporting the other criteria, beyond the earlier prefix.

Apply the same frozen technical relevance verifier to each
additional excerpt. Record individual positive checks and
candidate bundles that fill all missing criteria. Separately
count whether the cited spans include an exact-name inventor
list or a profile clause that explicitly names the candidate
and describes an action. These are detectable attribution cues.
Only a human review of the original evidence can verify whose
specific work the technical claim describes.

The arms share a check ceiling. Role-only and lexical policies
can have fewer eligible selections; report actual counts rather
than treating the realized work as equal. Recovery timing covers
selection and model calls on the eligible slice. The shared
primary pass, queues and transport are outside this timing.

For the qualitative diagnostic, retain the highest-scoring
checked quote per policy and query even when the model declines
every quote. Pair its evidence with the same candidate's
original partial evidence. Build two method-blind forms from
immutable source spans. This targeted packet helps diagnose
verifier misses and provides no frequency estimate for random
quotes. Keep all forms and source data restricted.

## Results

Jobs `43496` and `43499` reproduced the technical outcome
counts on one RTX 5090. Fourteen of 80 development requests
are eligible, with 32 initially partial candidate slots and a
total cap of 112 new checks. One request has an explicit
same-object cue.

| Source policy | Actual checks | Checks yielding at least one newly supported criterion | Completed candidate bundles | Recovery P95 |
| --- | ---: | ---: | ---: | ---: |
| Existing role windows | 5 | 0 | 0 | 8.3 ms |
| Lexical search across owned sources | 46 | 0 | 0 | 48.5 ms |
| Dense search across owned sources | 82 | 0 | 0 | 60.1 ms |

Every score is below or at zero. One of 82 dense checks
has a margin in [−1, 0]; the other checks fall below −1.
No verifier input was truncated. The expanded catalog
creates many more chances to read original text, but the
tested selector and verifier do not complete any bundle
for these cases. The observations cannot distinguish an
unavailable witness from a missed excerpt or a false
negative by the verifier.

Freeze the highest-scoring excerpt choices and corresponding
original-evidence controls for blind review. The private
packet contains 40 distinct evidence views on 13 requests
with two validated blank forms and zero human judgments.
The scope stays within development. No new recommendations
are issued and no content from the packet is released.
[Aggregate results](results.json)

## Reproduce

Use the pinned development inputs and model layout from the
[preceding qualifier-gap experiment](../enterprise-gap-bound-recovery-v1).
Run the benchmark on a Slurm CUDA compute node. Build its
review forms locally from the restricted trace.

```sh
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.enterprise_owned_gap_sources \
  DEVELOPMENT_QUERIES DEVELOPMENT_SOURCES ROLE_WINDOWS VECTORS_NPY VECTOR_METADATA \
  MODEL_ROOT NEW_AGGREGATES PRIVATE_TRACE
python -m experiments.retrieval.build_owned_gap_review \
  PRIVATE_DATASET PRIVATE_TRACE NEW_AGGREGATES NEW_PRIVATE_REVIEW_DIR
```
