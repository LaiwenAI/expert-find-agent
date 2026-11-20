# Cross Validated duplicate questions as expert and citation targets

This public pilot links each search question to a community-selected
canonical question, that question's accepted answer, and the
answer's original posting account. It adds an attributed evidence
target to answerer retrieval. The benchmark uses the
[Stack Exchange API](https://api.stackexchange.com/docs/advanced-search)
and the [original-question relation](https://api.stackexchange.com/docs/types/original-question).

Capture four 100-item pages of closed questions created in 2024,
ordered by creation date, and keep the ones explicitly closed as
duplicates with an available original-question relation. Capture
four pages of high-vote questions with accepted answers as
background. Include eligible canonical answers in the same fixed
catalog, so every retained target has a known witness. Require
both original question and answer text to have been created and
last edited before 2023. Validate the accepted-answer association
and a non-deleted owner. Remove exact title duplicates and queries
that directly link to the canonical question. Strip HTML markup and
URLs from indexed text. An additional audit found no direct links
to gold answer IDs in the retained query bodies.

The query input is the duplicate title and body. The evidence record
contains the canonical or background question title and body plus
the accepted answer text, attached to the answer account. The
manifest identifies the query IDs, accepted answer IDs and complete
candidate pool. No text, usernames or profiles are distributed in
this repository. Original authors, text, applicable licenses and
revision histories are available on Cross Validated at
`https://stats.stackexchange.com/q/<question-id>` and
`https://stats.stackexchange.com/a/<answer-id>`.

Compare BM25, a frozen dense encoder, and reciprocal-rank fusion
over the same catalog. Rank accounts by their highest-ranked
answer, then cite that answer. Report top-five account hits and
exact witness hits separately. The alternative gives each account
a second citation slot, or extends the candidate set to ten accounts
while citing only one answer per account. Those two alternatives use
at most ten slots. These are retrieved witnesses without a separate
reader. Paired intervals use connected components of shared gold
answer IDs as the sampling unit.

This is a fixed candidate-pool test with gold targets included,
and a single community with 2024 queries. The historical API
snapshot can observe accepted status after the content cutoff.
The official duplicate relation and answer acceptance are evidence
of an attributed answer, without judging the topical adequacy
of other outputs. All unlisted authors and answers remain
unjudged. The benchmark should not be reported as community-wide
recall, an independent assessment of answer correctness, or a
measure of enterprise recommendation quality.

## Results

Job `43693` ran on one RTX 5090. The panel contains 68 queries
in 67 shared-answer groups and 364 answers from 207 accounts.

| Retrieval method | Known account in top 5 | Known witness for top 5, one slot each | Exact witnesses per 340 ranked account slots | Local P95 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 47 | 38 | 39 | 0.69 ms |
| Qwen3-Embedding-0.6B | 63 | 59 | 64 | 7.76 ms |
| BM25 and dense RRF | 58 | 54 | 56 | 8.54 ms |

Dense adds 21 queries with exact known witnesses over BM25
(+30.88 points, descriptive paired 95% interval +19.40 to +42.65).
Finding the account alone misses the citation gap. Dense finds
the known account in 63 queries, four more than it can cite with
one slot each. Holding the dense top five accounts fixed and
adding a second citation slot increases witness hits from 59 to 62
with 567 realized citations. Expanding to ten accounts with
one citation each increases witness hits to 61 with 680 citations.
The realized budgets differ, and the two conditions are candidate
budget diagnostics. The scores do not establish that verifying
a second witness rather than another account improves selectivity.

The local query measurements include tokenization, retrieval,
and grouping answer owners after the catalog is built. The
dense run encodes the online query on GPU. Fusion time sums
sequential sparse and dense times and rank aggregation.
Offline dense catalog encoding took 2.43 seconds and lexical
index construction took 0.30 seconds. The workspace has the
original frozen API receipts and derived snapshot. No source text
or cache is published with the benchmark. [Aggregate results](results.json)

## Reproduce

The manifest pins the full capture and releases query/answer
relations and candidate IDs. The capture command reuses API
receipts already in the snapshot folder. Fresh API requests
may reflect revisions, votes or removals after this capture.
The evaluator checks the snapshot hash before scoring.

```sh
PYTHONPATH=src:. python -m experiments.community.crossvalidated_witness SNAPSHOT_DIR
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.community.evaluate_crossvalidated_witness \
    SNAPSHOT_DIR/snapshot.json SNAPSHOT_DIR/manifest.json QWEN_EMBEDDING NEW_RESULTS.json
```
