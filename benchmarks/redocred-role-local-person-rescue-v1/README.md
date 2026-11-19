# Recheck missing person candidates inside automatically routed sources

Reuse the public corpus, automatic source/NER index and frozen 4B
verifier from the
[automatic entity-index study](../redocred-automatic-entity-index-v1).
Hold the source list and its ordering fixed for every policy. Use the
same 161 exact work-name requests and 145 synthetic spelling swaps.
Candidate generation reads the raw source and spaCy spans. The
judgments are used only to score named candidates and diagnose misses.

| Policy | Proposal behavior |
| --- | --- |
| Automatic | The original predicted PERSON spans and existing within-document surname linking |
| Unified | Add capitalized multi-token runs within five tokens of the role cues; rank all candidates together |
| Guarded reserve | Preserve the first four slots and offer the fifth to a new capitalized name within three tokens of a role cue |
| Boundary reserve | Preserve the first four slots and offer the fifth to a new name or subspan from a predicted PERSON, ORG, PRODUCT, FAC or WORK_OF_ART span |

The name pattern allows a fixed list of lowercase connectors and
excludes common lead words. The boundary route enumerates capitalized
subspans of at most four tokens. It proposes single-token names only
from predicted PERSON regions. Candidate ranking uses the shared
work and role proximity scorer. Each policy proposes up to five
person–source pairs, including previously occupied and newly filled
slots. Report actual proposed counts as well as slot-normalized
scores. All arms use the same reader, up to two sentences per pair
and a numeric cutoff of 0.25. Query text and evidence contain the
proposed name, without any gold aliases at inference.

For the miss diagnostic, keep exact queries that retrieve a known
source but have no matching person anywhere in the automatic candidate
pool within the returned documents. Check whether any unique annotated
form for that person occurs inside a predicted entity span. Exact
type matches can overlap across the same query. A string contained in
a span does not establish a correct identity. The boundary policy was
designed after inspecting this diagnostic on the reused panel and has
an exploratory interpretation.

## Results

RTX 5090 job 43056 reproduces the main results from 43053 and adds
the boundary control. It also reproduces the fully automatic
decisions from the previous entity-index benchmark. CPU and GPU
retrieval counts agree.

| Query | Policy | Verified pairs | Top-5 known positives | Accepted known positive / rival / unjudged | P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| exact (161) | Automatic | 747 | 152 | 98 / 3 / 10 | 58.20 ms |
| exact (161) | Unified | 760 | 153 | 100 / 2 / 17 | 57.97 ms |
| exact (161) | Guarded reserve | 756 | 154 | 101 / 2 / 15 | 57.88 ms |
| exact (161) | Boundary reserve | 772 | 148 | 98 / 2 / 42 | 58.12 ms |
| transposition (145) | Automatic | 662 | 130 | 76 / 1 / 6 | 58.85 ms |
| transposition (145) | Unified | 679 | 129 | 78 / 1 / 10 | 58.33 ms |
| transposition (145) | Guarded reserve | 673 | 130 | 78 / 1 / 8 | 58.23 ms |
| transposition (145) | Boundary reserve | 690 | 122 | 75 / 1 / 32 | 58.38 ms |

In 25 exact queries that returned a known source but lacked the
annotated person in their candidate pool, 16 had a unique annotated
name form contained in a predicted entity span. The broader
boundary generator makes a known name available in 11 of the 25
misses. The narrower role-local rule does so in three. Availability
only indicates that one eligible surface form is in the pool,
before competition for the fifth slot.

The guarded reserve yields small positive-count changes (98 to 101
and 76 to 78) and more unjudged acceptances. Its paired intervals
include zero. The boundary route does not increase known positives
on exact inputs and accepts 42 unjudged pairs, versus 10 for the
baseline. On transpositions, the corresponding unjudged counts are
32 and six. Unjudged pairs may contain relevant people and require
independent adjudication. Broader proposal coverage alone does not
demonstrate reliable ranking of person evidence under a five-slot
verification budget. P95 differences are descriptive.

[Aggregate results](results.json)

## Reproduce

Restore the published Re-DocRED snapshots and the CPU NER projection
using the [automatic-index instructions](../redocred-automatic-entity-index-v1).
Run the 4B path in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python -m experiments.docred.role_local_person_rescue REDOCRED_PATH NER_PROJECTION CPU_RESULTS.json
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.role_local_person_rescue REDOCRED_PATH NER_PROJECTION GPU_RESULTS.json \
    --gpu-model PINNED_QWEN4B
```
