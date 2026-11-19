# Transfer a person-specific lexical gate to publisher contribution notes

A cheap ambiguity gate saved calls on
[Re-DocRED](../redocred-role-routing-v1). This test asks whether its
numeric thresholds transfer to another evidence format. Use openly
licensed PLOS ONE contribution notes from the original and
[independent captures](../plos-independent-cascade-v1). Verify every
XML checksum, require disjoint DOI inventories and exclude transfer
papers that share normalized byline names with either earlier split.
Build one deterministic single-role contrast per paper. The supported
initials belong to an author explicitly credited with the task in a
recognized publisher clause. The rival is credited with another task
on the same paper and absent from the target clause. Initials must
identify a unique author in the byline. The original computing panel
serves as development. Its energy test panel contributes only names
and DOIs to the exclusion rules.

A fixed, documented PLOS cue bank describes the five legacy tasks.
Find two sentences by maximizing combined lexical coverage of both
initials and the task cues. All short-window policies see the same
sentences and use the same author query and frozen 4B scoring prompt.
As in the source experiment, an exact within-sentence name–cue distance
provides a proximity score. The scientific source is fixed, so there
is no title matching signal. No gold role locations enter the window
selector or gate. Role names supplied in the query are paraphrased using
a pre-existing task wording bank.

| Policy | Calibration and input |
| --- | --- |
| Frozen reader | Re-DocRED standalone 4B cutoff, two sentences |
| Frozen gate | Re-DocRED lexical 75% band's two bounds and reader cutoff, two sentences |
| Frozen band, local cutoff | Re-DocRED lexical band with the PLOS short-reader cutoff |
| Local reader | PLOS development cutoff, two sentences |
| Local gate | Same lexical gate family with bounds and cutoff fit on PLOS development, two sentences |
| Local full reader | PLOS development cutoff, full note |

The transfer uses a domain-specific task vocabulary and query template
for all arms. The frozen comparison tests portability of the numeric
decision policy under those adaptations. The local cutoffs permit at
most 5% observed rival acceptance on development. Fit the local gate
with at most 75% 4B calls, optimizing both-correct decisions with the
same tie breakers as the Re-DocRED experiment. Fit all local settings
on the original computing panel and then freeze them.

One 4B call scores both people in a single batch. Rotate method order
and counterbalance candidate order. Timing includes live window
selection, lexical scoring and model calls, after calibration runs have
warmed the model. It excludes reading XML, searching sources, model load
and transport. Descriptive 95% intervals resample connected groups of
shared normalized author names. The benchmark reuses published source
captures under a new single-duty protocol. Its labels describe explicit
publisher credit inside the source and are limited to recognizable
legacy clauses.

## Results

RTX 5090 job 42997 reproduced the original policy decisions of job
42993 and added the frozen-band ablation. Development contains 32
papers across 31 author groups. Test contains 46 papers across 45
groups. The local calibration ceiling is one rival acceptance.

| Policy | Supported accepted / 46 | Rivals accepted / 46 | Both correct / 46 | 4B calls | Mean | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen reader | 43 | 13 | 30 | 46 | 16.37 ms | 16.97 ms |
| Frozen gate | 45 | 35 | 11 | 44 | 15.69 ms | 16.97 ms |
| Frozen band, local cutoff | 42 | 8 | 34 | 44 | 15.69 ms | 17.00 ms |
| Local reader | 42 | 9 | 33 | 46 | 16.35 ms | 16.96 ms |
| Local gate | 46 | 2 | 44 | 0 | 0.10 ms | 0.12 ms |
| Local full reader | 36 | 0 | 36 | 46 | 21.02 ms | 24.96 ms |

The frozen gate accepts 35 rivals on the 46-paper test, all on the 4B
fallback. The frozen band sends 44 of 46 pairs to the reader. Holding
those routed pairs fixed and substituting the PLOS reader cutoff reduces
rival acceptances from 35 to eight. This separates the effect of the
reader's transferred cutoff from the cost of the transferred lexical
band. The band alone does little to reduce calls on this format.

The local gate selects a zero-width ambiguity interval. It accepts any
positive lexical proximity, without calling the model. On test it
accepts all 46 supported authors and two rivals. The difference from
the local short reader in both-correct outcomes is +23.91 points
(descriptive author-group 95% interval +12.77 to +36.17).
The local full reader accepts no rivals in this sample, with ten fewer
supported authors than the lexical gate. The local short reader's
nine rival acceptances also exceed its one-rival development ceiling
in proportion.

The local lexical result reflects the regular role headers and
author-initial lists used to construct this panel. It serves as an
inexpensive control for recognizable publisher notes. A test of
expertise in narrative evidence needs separately adjudicated person
and duty links, including nearby mentions belonging to other roles.
The deployment implication is to scope a cutoff to its evidence
format and measure unsupported acceptances on a fresh panel before
trusting it.

[Aggregate results](results.json)

## Reproduce

Restore the two pinned publisher captures from the
[independent-capture instructions](../plos-independent-cascade-v1).
Use the exact [Re-DocRED routing result](../redocred-role-routing-v1/results.json)
and the same pinned Qwen3-4B-Instruct-2507 checkpoint.

```sh
PYTHONPATH=src:. python -m experiments.credit.legacy_role_route_transfer build \
  ORIGINAL_SNAPSHOT INDEPENDENT_SNAPSHOT PANEL.json
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.credit.legacy_role_route_transfer run \
    PANEL.json REDOCRED_ROUTING_RESULTS.json QWEN_READER NEW_RESULTS.json
```
