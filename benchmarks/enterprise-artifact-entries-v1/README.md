# Index role-bearing project and patent entries

This comparison replays the same 80 original enterprise development
requests from the [two-window recovery study](../enterprise-guarded-evidence-bundle-v1).
The initial search, parser, checkpoints, source corpus and verifier
stay fixed. On initial abstention, the alternative runs only when the
request has multiple criteria and its baseline already reads two windows.
It replaces those two windows with a same-owner pair. The four arms
have identical recovery verifier-pair counts on each request.

| Arm | Candidate and evidence allocation |
| --- | --- |
| Lexical windows | Original lexical role-window recovery |
| Dense windows | Previous dense embedding and same-owner pairing of disjoint windows |
| Dense entries | Same-owner pairing over role-bearing entries ranked by dense similarity separately for each criterion |
| Sparse entries | Same pairing over the same entries using deterministic IDF-weighted term overlap and no extra query embedding |

## Entry projection

Locate each role-bearing excerpt in its original profile field and
retain the containing line and exact original offsets. The original role
cue must lie inside the line. Lines with multiple indexed cues, excessive
length or a detected negation are excluded. The inferred unit is a
profile entry, with no claim that it independently corroborates a role.

The first projection deduplicates project lines within an owner using
the normalized first pre-role column. In this corpus, that column often
contains an agency or funding program. The second projection prefers
the normalized text of an explicit project locator. Otherwise it selects
the longest pre-role text column with at least six letters and fewer than
16 occurrences in role windows. The frequency cutoff is an exploratory
heuristic chosen without query outcomes. Patent lines require an explicit
number and the original named-inventor cue. Ordinary paper abstracts
remain in the shared initial corpus. They lack a named contribution
statement in this projection and enter neither entry variant. Project
titles and profile owner IDs are provenance hints. The projection can
still conflate records or leave aliases unresolved.

Each variant keeps up to 256 ranked candidates per criterion. A pair must
cover every criterion somewhere in its candidate lists. The allocator
minimizes the worst criterion rank and then the sum of criterion ranks,
with deterministic ties. The same verifier checks every criterion
against each of the two evidence windows. A complete single-window
witness takes priority. Otherwise a valid combination requires positive
model margins for all criteria across the pair. When no pair can be
allocated, the arm uses the original lexical recovery.

## Results

Both jobs in each Slurm array reproduce all non-timing outputs. The
two conventional window arms also reproduce the prior two-window
experiment. All four arms make 184 recovery verifier pairs across
80 requests. Each alternative successfully allocates a pair on all
23 eligible requests.

| Arm | Issued requests / 80 | Recommendation slots | Selection P95 (ms) | Serial first-query P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Lexical windows | 37 | 73 | — | 1062–1080 |
| Dense windows | 39 | 75 | 13.1–15.3 | 1076–1096 |
| Dense entries | 39 | 75 | 13.3–13.9 | 1078–1093 |
| Sparse entries | 37 | 73 | 2.4–3.2 | 1067–1085 |

Ranges cover both execution orders in both configurations. Selection
timing covers the 23 eligible requests and includes the extra query
embedding in the dense arms. Serial timing covers all 80 requests and
excludes offline index construction and service queues.

The more specific projection retains 4,745 entries for 1,015 owners
versus 4,540 entries for 979 owners with the first-column key. It flags
81 same-owner duplicates versus 444. The two dense entry arms issue
identical recommendations and citations. Against dense windows, they
keep every expert and replace one exact citation view. The possible
supported P@5 change on this trace is −0.25 to +0.25 percentage points,
over 400 recommendation slots. The citation has not been judged.

Both sparse arms reproduce the exact lexical baseline outputs. They
miss the two provisional candidates recovered by the dense arms.
Thus the allocation speed-up does not preserve the additional
outputs, and the expanded entry inventory supplies no new expert
relative to dense windows in this development replay. The private
review packet contains 76 distinct exact views and no judgments.

## Reproduce

On an allocated Slurm GPU, restore the restricted source projection,
role windows, cache and packed inference reference from the
[previous study](../enterprise-guarded-evidence-bundle-v1#reproduce).
Use distinct output paths for each run and repeat with `--reverse`
to change the first arm in the per-request rotation.

```sh
export PYTHONPATH=src:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python -m experiments.retrieval.enterprise_artifact_entries \
  QUERIES CORPUS ROLE_WINDOWS VECTORS VECTOR_META MODELS PACKED_REFERENCE \
  SUMMARY PRIVATE_TRACE --title-key specific
```

Use `--title-key first_column` for the simpler projection. The
[aggregate report](results.json) pins the exact code, model revisions,
input fingerprints and execution records. Keep exact-view reviews and
per-request traces private.
