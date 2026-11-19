# Search for two same-person duties across a structured publisher catalog

The earlier [open publisher-note experiments](../plos-open-role-scope-v1)
used legacy initials and concentrated on duty interpretation. This study
measures retrieval, source confirmation, and same-person binding against
the publisher's modern structured contribution declarations.

Combine the pinned [130-paper source inventory](../plos-credit-cross-source-v1/manifest.json)
and [external PLOS capture](../plos-credit-external-v1/manifest.json), merging
duplicate DOIs only when the indexed text and author declarations agree.
Verify every raw XML digest. The searchable catalog has 159 unique papers
with usable title and abstract splits. Hold out the last two abstract
sentences from the primary index. An exact-study request supplies these
sentences or up to eight derived keywords and names two duties explicitly.
Calibration uses 24 articles selected by a fixed DOI hash from the original
inventory. Evaluation uses all 24 articles admitted by the earlier external
manifest's DOI and author-identity exclusions.

For each evaluation paper, select one pair of duties credited to at least one
but fewer than all authors. Also select, when available, a pair whose duties
both occur in the article but are never credited to the same author. Replay
both views with the intended source present and withheld. The expected
answer is an exact study and the full set of qualifying author ordinals
within it. A different article cannot satisfy the exact-study request.

All policies rank the same complete catalog using reciprocal-rank fusion of
lexical overlap and exact dense scores. They spend one live query embedding
and at most four source confirmations. Reuse the original Qwen3-Embedding
checkpoint. Each route first attempts to confirm the highest ranked source
without using author roles. If it confirms, check only that study, even if
no author qualifies. If it fails, the four-source control checks the next
three sources in global rank order. The postings route checks the first
three remaining sources with a declared same-author match. The body variants
apply the same fallback sequence and may confirm each nominated article by
its best lexical body window if its dense score is below threshold. Remove
body sentences near-duplicating the held abstract excerpt before indexing.

The dense threshold is the maximum top-nominee score when the development
source is withheld. The body threshold is the maximal body overlap among
the first four distinct false sources from that calibration that fail the
dense threshold. Thresholds are frozen per clue and shared by all arms.
This calibration constrains observed errors in development and does not
define a population risk guarantee.

## Results

Slurm jobs 42910 and 42913 agreed on every outcome and inspection count for
the dense-only arms. Job 42913 ran on an RTX 5090 and includes the body
comparisons. The table gives clean successes among 24 supported
source-present requests per clue. False sources from the withheld view are
reported separately for the supported queries. Every policy had zero
wrong-author returns on a confirmed correct source. That reflects direct
use of the publisher's role fields.

| Route | Excerpt clean | Excerpt withheld wrong source | Excerpt P95 | Keywords clean | Keywords withheld wrong source | Keywords P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense top one | 11 | 0 | 7.85 ms | 11 | 1 | 7.28 ms |
| Dense top four | 12 | 0 | 7.85 ms | 11 | 1 | 7.28 ms |
| Dense with owner postings | 12 | 0 | 7.85 ms | 11 | 1 | 7.29 ms |
| Body top one | 17 | 1 | 8.08 ms | 15 | 2 | 7.35 ms |
| Body top four | 19 | 1 | 8.52 ms | 15 | 2 | 7.34 ms |
| Body with owner postings | 20 | 1 | 8.57 ms | 15 | 2 | 7.34 ms |

Same-author postings add one clean excerpt request over body top four, using
26 rather than 27 body checks in the 24 requests. Keywords have no clean
gain. For the 19 split-author controls, no route emits an unsupported author
from the correct paper. With the source withheld in those controls, the
keyword body-postings route still emits three wrong-source recommendations,
versus two for body top four. Body confirmation accounts for most of the
clean gains and also admits additional wrong sources. Source verification
is the larger bottleneck in this slice.

Timing includes live encoding, full catalog retrieval, conditional body
checks, and direct author binding. It excludes indexing, queues, and
transport. The task has explicit duties and an exact-study target. It
does not measure natural-language role inference or enterprise expertise.
The excerpt-derived keywords and supported/withheld/split views are paired.

[Aggregate results](results.json)

## Reproduce

Use the exact CC BY publisher snapshots listed in both manifests. The runner
accepts multiple raw XML directories and resolves each file by DOI suffix
and SHA-256. Missing files can be requested again using the URLs in the
manifests, retaining only byte-identical copies. Run in a Slurm CUDA job.

```sh
PYTHONPATH=src:. python -m experiments.credit.open_person_postings \
  PRIOR_MANIFEST EXTERNAL_MANIFEST EMBEDDING_CHECKPOINT NEW_RESULTS.json \
  PRIOR_XML_DIRECTORY EXTERNAL_XML_DIRECTORY [ADDITIONAL_XML_DIRECTORY...]
```
