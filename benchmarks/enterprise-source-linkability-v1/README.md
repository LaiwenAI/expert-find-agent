# Audit the availability of source locators for enterprise attribution

Use the original profile values and owners from the private,
pinned 52,667-source enterprise development inventory. Use the
[existing declaration-cue parser](../enterprise-declaration-reach-v1).
For project fields, detect project, grant, funding and work-item
identifier labels followed by a number-shaped token. For patent
fields, detect a jurisdiction-prefixed number or explicitly
labeled patent/application/publication number. Keep original
character offsets. This is a syntactic screen, with no live
registry lookup or validation of identifiers or records.

Count separately a locator-shaped span, a named role cue,
the two in a single profile field, and the two in the same
profile clause. A patent name cue requires an exact normalized
name in the parsed inventor list. A project cue requires an
exact normalized owner string in a role-bearing clause. It
can still mention the owner in another grammatical capacity.
Neither cue proves registry accuracy, personal execution,
roster completeness, or suitability for a requirement.

Rerun the [25-request headroom search](../enterprise-joint-headroom-v1)
with the same hybrid ordering, source excerpt selection and
0.6B verifier. For each candidate, record whether any owned
catalog source carries both cues in one clause and whether the
inspected excerpts include that exact clause. Also record its
intersection with model topic support and the emitted provisional
citations. Do not change any candidate, evidence selection or
verifier decision. The 25 requests are a development diagnostic
selected for multiple parsed conditions and original abstention.

Report aggregates only. Restricted profile data, source
identifiers, source text, citation offsets and individual
request outcomes stay in the authorized workspace. No identifiers
are sent to outside services.

## Results

The development fields include 1,832 project records and 713
patent records. Paper, paper-list, and other profile fields
are outside the locator scan.

| Field | Locator shape | Named role cue | Both in same field | Both in same clause |
| --- | ---: | ---: | ---: | ---: |
| Project (1,832) | 63 | 21 | 1 | 1 |
| Patent (713) | 504 | 9 | 7 | 5 |

Slurm tasks in array `45074` on the `ws3` RTX 5090 preserve the
headroom benchmark outcomes. Among 25 requests, only one has
a top-20 candidate with a co-located cue anywhere in the
available catalog. That same request already has a matching
excerpt in the top-five, two-passage baseline.

| Fixed shortlist and read depth | Inspected excerpts | Locator in excerpt | Named role cue in excerpt | Both in same clause in excerpt | Requests with both in any inspected excerpt |
| --- | ---: | ---: | ---: | ---: | ---: |
| Five people, two passages | 250 | 34 | 2 | 1 | 1 |
| Five people, up to eight passages | 869 | 54 | 11 | 4 | 1 |
| Twenty people, two passages | 997 | 118 | 3 | 1 | 1 |
| Twenty people, up to eight passages | 3,344 | 192 | 25 | 4 | 1 |

Expanded views can repeat an original source and offsets, so four
matching excerpts do not imply four distinct corroborating records.
The widest arm has one candidate with both model topic support
and a co-located cue. None of its seven provisional candidates
cites such a clause in the winning evidence. Topic support and the
named locator can come from different passages.

The main bottleneck for this screening rule is the availability
of a linkable named clause among retrieved candidates. Deeper
reading within those candidates reveals more locator-shaped text
and does not expand request coverage. This evidence supports
targeted enrichment where a source can be verified. It provides
no basis to exclude unlinked projects or patents from search.
[Aggregate corpus counts and replay results](results.json).

## Reproduce

Prepare the restricted projection from the original development
export. Run the replay using the pinned headroom inputs and the
same Slurm GPU configuration as that experiment.

```sh
PYTHONPATH=src:. python -m experiments.retrieval.enterprise_source_linkability \
  PRIVATE_DATASET DEVELOPMENT_CORPUS PRIVATE_PROJECTION

PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.enterprise_joint_headroom \
  QUERIES CORPUS ROLE_WINDOWS VECTORS VECTOR_META MODELS SCRATCH \
  FIRST_ROUND_TRACE FIRST_ROUND_AGGREGATE NEW_AGGREGATE NEW_PRIVATE_TRACE \
  --linkability PRIVATE_PROJECTION
```

Repeat with `--order-offset 1` to reverse request order.
