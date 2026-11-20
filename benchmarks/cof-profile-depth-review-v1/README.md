# Blind review of evidence that source-depth scanning does and does not recover

Replay the [complete-profile diagnostic](../cof-full-profile-support-v1).
Freeze the two panel references and require exact agreement on
cohort and label hashes, corpus sizes, model revisions, audit
shape, every outcome count and all truncation counts. Preserve
the original dense ranking, verifier, decisions and indexed
catalog. Only inspect the 72 released positive cases among
the top-five nominees. This set cannot estimate precision or
adverse risk over all candidates.

Include all profiles that fail the two-source requirement at
depth three. These include profiles rescued later, profiles
with only one model-accepted source anywhere, and profiles
with none. Include six deterministic hashed controls per panel
from profiles already supported at depth three.

For each included profile, show all three initial sources and
the first two model-accepted sources anywhere in the catalog,
deduplicated. For cases with fewer than two model-accepted
sources, include the highest-scoring rejected source outside
the first three. This is an evidence-enriched review sample.
A missing claim in the sampled sources does not establish that
the full profile or the expert lacks it.

Reviewers see the request, the cited title and abstract and
up to three distinct other profile titles as identity anchors.
Titles matching any displayed citation are excluded. The
packet hides ranks, margins, actions and strata. It separates
three judgments:

1. Support in the indexed title and abstract.
2. Support in the original full text, with access failure
   recorded separately.
3. A verbatim original-source contribution statement and
   independent identity links connecting the candidate to the
   relevant claim.

Collect two independently attributed forms. The scorer
verifies the immutable fields, exact abstract quotations
for definite content judgments, and the required evidence
fields for full-text and contribution judgments. Reviewers
assess the semantics and correctness of those external quotes.
Report agreement, insufficiency, missing judgments, and
disputes by sampled stratum, depth and verifier decision.
These are diagnostics for separating evidence-window limits
from potential verifier misses. The benchmark does not infer
personal expertise from paper-level relevance alone.

## Source identity sensitivity

On all top-five known positives, tabulate the original
source-record count alongside title-and-abstract and
title-only uniqueness criteria. Normalize punctuation,
case and Unicode variation. For each depth, apply
deduplication within the original prefix. No backfill or
verifier change is allowed in this diagnostic. A matching
title is a conservative proxy for publication identity.
Only a verified link to the original publication can
adjudicate versions, renamed variants and homonyms.

## Verified packet and source-identity results

Both RTX 5090 tasks in Slurm array 45335 reproduced the
published diagnostic counts. The 16 KDD and 30 SciRepEval
candidate–query profiles yield 63 and 114 blind citation
cases. Every blank form passed the integrity check. These
are ready for independent assessment, with zero human
judgments at export.

| Panel | Depth | Two accepted records | Two distinct normalized texts | Two distinct normalized titles |
| --- | --- | ---: | ---: | ---: |
| KDD (23 known positives) | 3 | 13 | 13 | 9 |
| KDD | 12 | 13 | 13 | 10 |
| KDD | Full history | 17 | 17 | 15 |
| SciRepEval (49 known positives) | 3 | 25 | 25 | 24 |
| SciRepEval | 12 | 31 | 31 | 30 |
| SciRepEval | Full history | 31 | 31 | 30 |

By the title proxy on the combined 72 candidates, the
counts change from 38 to 33 at depth three and from
48 to 45 after reading all sources. This comparison
preserves the original prefix and every verifier verdict.
It measures source-identity sensitivity, without a quality
or latency comparison of a deduplicating serving policy.

Two sampled KDD title groups link to distinct conference
and journal DOIs for related versions. The authors identify
the two versions of *Graph Summarization with Quality
Guarantees* in the [project repository](https://github.com/rionda/graphsumm).
The journal version of *Fast Approximation of Betweenness
Centrality through Sampling* explicitly identifies the
conference version in its introduction. [Journal manuscript](https://matteo.rionda.to/papers/RiondatoKornaropoulos-BetweennessSampling-DMKD.pdf).
Both examples show why distinct record IDs or even DOIs
are insufficient to establish independent evidence.
Matching titles alone also cannot establish that every
matched pair represents one line of work.

[KDD packet summary](kdd.json) · [SciRepEval packet summary](scirepeval.json)

## Reproduce

Create the two packets on Slurm CUDA from the same checkpoints
as the full-profile diagnostic. Keep raw citation text, private
case mappings and reviewer forms outside the public checkout.
Give each reviewer only the form and its review instructions.
The comparison can run on a local CPU.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.retrieval.cof_profile_depth_review export \
  KDD PUBLISHED_AUDIT_JSON QWEN06B_CHECKPOINT RERANKER_CHECKPOINT NEW_PACKET

PYTHONPATH=src:. python -m experiments.retrieval.cof_profile_depth_review compare \
  PACKET REVIEWER_ONE_JSONL REVIEWER_TWO_JSONL NEW_AGGREGATE
```
