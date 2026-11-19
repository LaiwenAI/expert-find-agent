# Retrieve role evidence from the complete Re-DocRED development and test collection

Index all 1,000 documents in the pinned public
[Re-DocRED snapshot](https://aclanthology.org/2022.emnlp-main.580/).
For each work and relation with a positive and a competing-role person
in at least one test article, query the canonical work name and role.
No candidate names, target document IDs, relation labels, or human
evidence locations enter retrieval. The documents' published mention
annotations supply normalized work aliases, occurrence positions and
candidate person entities. This assumes typed mentions are available,
and query work names come from that vocabulary. The benchmark contains
161 work–role requests from 41 reference articles. It is a collection
search for role credit, with no claim about general expertise.

Form judgments separately from the retrieval index using the revised
relation labels. A known positive is a particular person credited for
the queried work and role in a particular document. A known competing
candidate has a different labeled role for the same work in a document
that also contains a positive. Other returned person–document pairs
remain unjudged. Names map to local source identities and are not
merged into a single cross-document person. One query has multiple
positive documents. The primary positive count uses all five ranked
slots in its denominator, including slots left empty by the policy.

| Document routing | Search policy |
| --- | --- |
| Flat four | BM25 over the corpus with work and role terms; take up to four documents |
| Alias-boost four | Full BM25 with exact indexed work mentions above other matches; fill up to four |
| Alias-route four | Score only the work's exact mention postings when available, with the same four-document cap; use flat BM25 on a missing alias |

Every route ranks people using the same maximum mention proximity to the
role cues and to the work mention, plus a small document-order prior.
Person scoring uses pre-indexed word/mention positions and never uses
relationship labels. The comparison against alias boost holds exact
source prioritization constant and tests omitting lexical-only
backfill. All 161 test queries have an exact anchor, so these metrics
describe that branch. They do not establish behavior on missing or
noisy work names. Timing covers live lookup and person ranking over
the resident indexes. Report median per request over three warmed
repetitions, then the P50 and P95 across requests.

A second pass independently runs search and the same frozen
Qwen3-4B-Instruct-2507 on up to five person–source pairs per method.
Select two sentences by lexical coverage of the work, candidate name and
role cues. Score explicit support and abstain below the cutoff. Fit each
cutoff to at most 5% observed rival acceptances on the development
requests, removing any development query whose work name occurs in the
test set. Development source scoring uses labels only for development
documents. Include the same corpus for retrieval in both stages.
Report accepted unjudged candidates separately. The cutoff conditions on
available competing-role labels; it does not bound error over all
returned pairs. GPU timings include the live search, window selection,
prompt preparation and model inference under warm, sequential replay.

## Live retrieval results

| Policy | Opened documents | Posting entries inspected | Person occurrences scored | Positive source present / 161 | Known positives in Top-5 | Known rivals in Top-5 | Unjudged Top-5 | CPU P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat four | 644 | 116,600 | 3,518 | 146 | 177 | 144 | 484 | 0.251 ms |
| Alias-boost four | 644 | 116,600 | 3,548 | 161 | 192 | 176 | 437 | 0.250 ms |
| Alias-route four | 165 | 1,177 | 1,136 | 161 | 222 | 254 | 197 | 0.022 ms |

The alias route retains all 161 known-source hits of alias boost and
reduces actual document expansions by 74.4%. The two routes share exact
alias prioritization, but only the routed method avoids scanning the
remaining posting entries. Its positive P@5 difference on the 805
fixed slots is +3.73 percentage points (document-clustered 95%
interval +2.14 to +5.29). Known competing candidates also increase by 78.
The source filter concentrates the candidate list on the intended work
and still requires person-level verification.

## Search and role verification on RTX 5090

Slurm job 43034 reproduced the main decisions of 43030. It adds two
threshold controls. There are 172 work-disjoint development requests.
The table applies the per-route 5% observed development-rival cap.

| Route | Submitted pairs | Accepted positives | Accepted competitors | Accepted unjudged | Queries retrieving an accepted positive / 161 | Component P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat four | 805 | 126 | 4 | 10 | 91 | 57.69 ms |
| Alias-boost four | 805 | 118 | 3 | 9 | 91 | 57.59 ms |
| Alias-route four | 673 | 165 | 4 | 9 | 116 | 56.94 ms |

Alias routing sends 16.4% fewer person–source pairs to the reader than
either control. The difference against alias boost in accepted positive
P@5 is +5.84 percentage points, with a document-clustered 95% interval
from +4.05 to +7.73. The P95 difference is under one millisecond.
The counts show positive role instances according to document-level
labels. They do not establish that the selected two-sentence excerpt
entails the relation for every accepted row.

The two sensitivity checks reduce the positive-count advantage. Under
the same numeric threshold (the alias-boost cutoff), flat, alias boost,
and alias route accept 107, 118 and 136 positives, respectively, and
three annotated rivals each. Under the same absolute calibration
budget of eight rival acceptances, the corresponding positive counts
are 126, 111 and 131, with four, three and three annotated rivals.
Those threshold controls were added after the first run and are
descriptive on the reused panel. The routed policy retains an advantage
on these exact-anchor queries. Performance under noisy work names,
automatically extracted mentions, and broader enterprise suitability
remains open.

[Aggregate results](results.json)

## Reproduce

Use the same dev and test files pinned by the
[original Re-DocRED person-role benchmark](../redocred-person-evidence-v1).
Run the resident index on CPU. The verification comparison uses the
pinned open reader in a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python -m experiments.docred.global_role_search REDOCRED_DIRECTORY INDEX_RESULTS.json
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.docred.global_role_verify \
    REDOCRED_DIRECTORY PINNED_QWEN4B VERIFICATION_RESULTS.json
```
