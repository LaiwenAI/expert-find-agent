# Measure role-normalization loss on two-topic prose requests

Take the 160 PLOS and 51 eLife heading requests from the
[joint source benchmark](../live-joint-source-request-v1).
Replace the role fields with descriptions from the pinned
[NISO examples](https://zenodo.org/records/18421449).
Hold out role-specific descriptions with the same deterministic
phrase split used in the earlier live bundles study. Choose
a held-out phrase per request and role. The same phrase may be
used for more than one request. The role projection is trained on
84 separate task descriptions whose normalized strings share no
exact match with the NISO examples.

Build two controlled prose templates. One quotes the topics and
puts the role description after each topic. The other puts the
unquoted topic after each role description. Both templates request
two documented contributions by the same person. Neither names
papers, a person or canonical role fields. Parse using deterministic
rules for those two forms. This evaluates role interpretation and
the full downstream path within controlled boundaries. Open-ended
segmentation and complex qualification language require additional data.

| Arm | Translate prose into two role labels |
| --- | --- |
| Known roles | Insert the intended canonical roles after the template parse |
| Ridge | Use the frozen 0.6B encoder and the ridge role head |
| Fused ridge | Encode both topics and both role descriptions in one forward, then apply the same ridge head |
| Reader | Use a frozen Qwen3-4B chat reader with the full CRediT option list |
| Guarded | Use the ridge head and invoke the 4B reader on low-gap role decisions |

Fit the guarded margin cutoff on NISO calibration phrases, before
either publisher test, using the same quantile grid as the prior
routing benchmark. The source path then uses the same live hybrid
retrieval, owner-postings nomination and three local declaration
reads. It returns at most one matching person and two source IDs.
Topic vectors in the fused arm use the same textual instruction,
but encounter a different padding and physical batch shape.

Report requests with both topics and both roles recovered, issued
responses, returns missing a requested role in the cited contributor
declaration, and matches with
both intended roles and topic tags. A role-correct citation with an
unlisted topic remains unresolved. The evaluator does not treat a
missing publisher topic label as contradictory evidence. The two
prose forms share the same base queries and held role descriptions.

Warm local P95 includes input parsing, role classification,
query embedding, live nomination, file reads and JSON response.
It excludes network, queueing, index construction and startup.

## Results

Both RTX 5090 runs in Slurm array `44534` reproduce all
non-timing aggregates. The two prose forms give the same
non-timing results. They
share role descriptions and therefore do not count as
independent replication. Both derive all topic boundaries
in this controlled grammar. The table gives request-level
counts from both publishers; known matches require the
intended role and the publisher topic for both citations.

| Panel | Arm | Both roles correct | Known matches | Issued with a wrong declared role |
| --- | --- | ---: | ---: | ---: |
| PLOS, 160 | Known roles | 160 | 49 | 0 |
| | Ridge | 110 | 41 | 5 |
| | Fused ridge | 110 | 41 | 5 |
| | 4B reader | 57 | 31 | 5 |
| | Ridge with reader fallback | 101 | 40 | 5 |
| eLife headings, 51 | Known roles | 51 | 14 | 0 |
| | Ridge | 34 | 9 | 1 |
| | Fused ridge | 34 | 10 | 1 |
| | 4B reader | 20 | 11 | 1 |
| | Ridge with reader fallback | 33 | 8 | 1 |

The eLife fused and separate routes assign both roles correctly
on the same aggregate number of requests, with one additional
known fused result. Fusion changes the physical embedding
layout of topic inputs as well as the tasks, and does not
establish response equivalence. The 4B baseline improves the
known eLife count by one over fused ridge despite a lower
full-request normalization count. It costs more than the fused
path on both panels. The tested gap-triggered fallback does
not improve either publisher's known count.

## Reproduce

Restore the publisher snapshots and verified model checkpoints
from the earlier benchmarks. Regenerate the pinned NISO inventory
with the existing extractor. Run in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.prose_joint_source_request \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  QWEN_EMBEDDING QWEN4B NEW_RESULTS
```

Repeat with `--order-offset 1`.


The abstention diagnostic uses the same inputs except the 4B
checkpoint:

```sh
PYTHONPATH=src:. python -m experiments.credit.prose_role_gap_abstention   PLOS_XML benchmarks/plos-credit-cross-source-v1   ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json   NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json   QWEN_EMBEDDING NEW_RESULTS
```
