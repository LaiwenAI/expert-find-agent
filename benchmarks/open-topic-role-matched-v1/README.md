# Compare joint planning with a sequential route using the same role metadata

The [mixed-publisher study](../mixed-publisher-joint-v1) compared a
joint contributor-role plan against a byline-only plan and a sequential
gap repair. The [specific-publication control](../plos-matched-contributor-reads-v1)
found higher substitute risk when both planners had the same personal-role
metadata. This follow-up isolates the effect of joint planning on the
open-topic PLOS and eLife requests.

Freeze the 452-article mixed catalog, the 160 PLOS and 51 eLife topic-role
requests, and the top-128 hybrid pool per condition. Both routes see
the same known personal roles and role-unknown contributor flags.
Both read exactly three distinct local declaration files and use the
same final ranking and original declaration verifier. The sequential
route chooses its first two sources independently, preferring the top
source with the corresponding role on each side. After reading both
declarations, it uses the known role postings to select a third source
that could complete a contributor's proof. The joint route uses the
published two-lane plan and the same role inventory. Report any difference from the archived RTX 5090 joint result, and compare only arms run together.

Count publisher-tag-supported proofs, unresolved topical coverage, and
reachable tag-supported proofs after the three reads. Report natural
issuance, the shared set of issuing requests, and a panel-wide matched
quota using the same citation rank criterion. The latter is an offline
comparison. Cluster sampling intervals by topic pair. Treat an untagged
or cross-publisher proof as unresolved. For the paired change, give
the possible count interval if those unresolved topical labels can be
either supported or unsupported. Cancel identical evidence packages
across arms. This identification interval concerns missing judgments.
It is separate from the sampling interval over queries with the recorded
labels.

Warm local timing includes live query encoding, shared retrieval,
nomination, reads and response serialization. It excludes network
transport and index construction. Publisher tags and declaration labels
are structured measurements. The study reuses previous requests and does
not supply independent judgments of candidate suitability.

## Results

Both RTX 4090 runs in Slurm array 47401 reproduce the source plans,
issued proofs and aggregate outcomes. PLOS replays use 480 reads per
arm, and eLife uses 153. The joint route matches the archived RTX
5090 count of label-supported proofs in both panels. It returns
three fewer unresolved PLOS proofs in these runs.

| Panel and issuance policy | Role sequential, known / issued | Role joint, known / issued | Joint minus sequential possible difference with missing labels |
| --- | ---: | ---: | ---: |
| PLOS, natural | 30 / 45 | 85 / 140 | +48 to +102 |
| PLOS, common issuing requests | 30 / 45 | 30 / 45 | −7 to +7 |
| PLOS, equal panel-wide quota | 30 / 45 | 36 / 45 | −7 to +13 |
| eLife, natural | 0 / 3 | 46 / 49 | +43 to +49 |
| eLife, common issuing requests | 0 / 3 | 3 / 3 | 0 to +3 |
| eLife, equal panel-wide quota | 0 / 3 | 3 / 3 | 0 to +3 |

The common PLOS requests include 12 changed experts. Four changes
gain a tagged proof and four lose one. At the quota of 45, the paired
query-level difference with recorded tags is +3.75 percentage points,
with a topic-pair bootstrap interval of −6.90 to +14.37. Both methods
take roughly 18 ms at warm local P95. The natural-issuance gain is
robust under the specified missing-label bounds, and is largely a
coverage result. The matched comparison leaves the direction of the
PLOS quality difference unresolved. On eLife, the stringent matched
comparison is based on just three issuances per method.

[Aggregate results and replication hashes](results.json)

## Reproduce

Restore the [original mixed-catalog inputs](../mixed-publisher-joint-v1#reproduce)
and use fresh outputs on an allocated Slurm GPU. Repeat with
`--reverse`.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.credit.open_topic_role_matched \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json QWEN_EMBEDDING \
  benchmarks/mixed-publisher-joint-v1/results.json \
  benchmarks/live-joint-source-request-v1/results.json NEW_RESULTS
```

## Independent review at matched issuance

The primary review compares the equal-issuance quota on each panel.
The common requests answered by both routes give a secondary estimate
on that specific subset. Freeze selections before review. Show reviewers
only the requested topics and roles, original paper title and abstract,
publisher DOI and byline identity, and original contribution excerpts.
The source DOI gives access to the full article. Share only `review.html`
with reviewers. Keep the case-to-method key and the reviewers' records local.
Ask two reviewers to assess each topic, each role, and contributor identity
independently. An unsupported package requires a shared contradiction on
at least one dimension. Disagreement and inconclusive evidence remain
unresolved in the conservative difference bounds. An identical evidence
package receives the same judgment in both arms.

| Panel and selection | Issued per method | Identical evidence packages in both arms | Distinct review cards |
| --- | ---: | ---: | ---: |
| PLOS equal quota | 45 | 17 | 73 |
| PLOS common issuing requests | 45 | 33 | 57 |
| eLife equal quota | 3 | 0 | 6 |
| eLife common issuing requests | 3 | 0 | 6 |

The union contains 90 cards. Slurm array 47414 on ws6 regenerated the
original two decision signatures and produced byte-identical cards and
keys under reversed traversal. The reviewed outcome remains pending.
The eLife matched comparison is descriptive given its three issuances
per method.

To rebuild, append `--blind-review benchmarks/open-topic-role-matched-v1/results.json NEW_PRIVATE_DIRECTORY`
to the command above. The export checks both archived decision signatures
before producing the reviewer page. With two independent exports, score
using the existing scorer:

```sh
PYTHONPATH=src:. python -m experiments.credit.blind_joint_topic_review score \
  PRIVATE_DIRECTORY/key.json NEW_SCORE REVIEWER_1.json REVIEWER_2.json
```
