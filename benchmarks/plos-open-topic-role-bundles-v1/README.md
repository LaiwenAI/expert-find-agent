# Retrieve cross-publication proof from open topic-and-role requests

Use 130 publisher records from the pinned [PLOS contributor corpus](../plos-credit-cross-source-v1).
A request specifies two different topic–CRediT-role pairs and names
neither a person nor an article. Each known qualifying person must
have the requested role on a paper with the respective publisher topic,
for both conditions. Include positive requests for which no person has
a single article bearing both required topic-and-role assignments.
Every qualified candidate therefore requires two independent publisher
records. Use checksum-validated publisher ORCID links for identity.

Construct the reference from original contribution declarations and
the most specific terms of the publisher's discipline paths. Each topic
appears on at least four indexed papers. Consider eligible two-condition
requests for contributors with multiple publications, retain at most
12 hash-selected requests per anchor contributor, and deduplicate
across anchor contributors. Topics and roles are controlled labels.
All rankings use titles and abstracts, without publisher subject tags.
The document pool excludes correction and other notice categories.

## Matched read budget

Retrieve the top eight documents per topic with BM25, or reciprocal-rank
fusion of BM25 and a frozen Qwen3 0.6B embedding. Each route can attach
a document to a condition only when it is in that condition's pool.
A complete online result rechecks exact declared roles by author and
requires two distinct source identifiers. Report the strongest single
qualifying bundle by its weakest topic rank and then by summed ranks.
The actual publisher topic labels are revealed only during evaluation.
A returned citation with an unlisted topic remains unresolved.

| Policy | Allocate the read budget |
| --- | --- |
| Initial | Read the top two distinct articles, starting with one per condition |
| Parallel topic | Add a third by topical ranking without using initial declarations |
| Parallel byline | Add a third that shares an indexed contributor with initial documents across conditions |
| Pooled gap | Read initial declarations, then favor the condition with fewer observed eligible authors |
| Owner gap | Read initial declarations, then favor a source on the other condition for an author with incomplete evidence |
| Joint byline | Select a pair of complementary documents with a shared ORCID before reading, then choose one more that can form a pair with either |
| Joint role union | Add an article-wide role-presence filter to joint byline selection |

Joint role union uses materialized article-level role metadata offline.
The union cannot establish who held a role. Author-specific declarations
are checked only on selected articles for every route. The procedures
share the same candidate universe and each extra-read route may inspect
up to three distinct declarations. A missing lexical pool remains empty.

Count reference proofs available anywhere within the pools, proofs within
the two common initial articles, and proofs reachable after one additional
read from the same pools. These label-derived ceilings are never inputs to
a policy. Compare the known coverage, issued results with incomplete
topic labels, and descriptive topic-pair clustered intervals. The cohort
is enriched for cross-publication positives and reuses publishers and
contributors. An aggregate measures this snapshot's proxy labels, without
estimating the prevalence of such needs or end-to-end service quality.

## Results

Both tasks of Slurm array `44420` ran on RTX 5090 on ws5 and
reproduced all aggregates under separate Python hash seeds.
The 160 queries use 43 topic terms and 102 distinct topic pairs.
They have 177 reference person matches. The 8-per-topic pools
contain a complete reference bundle for 12 requests with BM25
and 23 with hybrid retrieval. At most five and 13 requests,
respectively, can reach a complete bundle if the first two
articles are irrevocably fixed and only one more can be read.

| Policy | BM25 known / unresolved | Hybrid known / unresolved | Hybrid reads |
| --- | ---: | ---: | ---: |
| Initial | 2 / 0 | 0 / 0 | 320 |
| Parallel topic | 2 / 0 | 1 / 0 | 480 |
| Parallel byline | 5 / 0 | 13 / 0 | 480 |
| Pooled gap | 3 / 0 | 1 / 0 | 480 |
| Owner gap | 5 / 0 | 13 / 0 | 480 |
| Joint byline | 11 / 0 | 22 / 1 | 480 |
| Joint role union | 11 / 0 | 22 / 1 | 480 |

For hybrid retrieval, the joint byline plan adds 21 known results
against the parallel topical schedule at equal reads (+13.13
percentage points per request, topic-pair clustered descriptive
95% interval +4.49 to +24.16). It recovers 22 of the 23 achievable
requests in those pools. The advantage comes from changing the
initial allocation: once the common top-two articles have been
fixed, the one-more oracle reaches only 13 requests. The static
parallel byline schedule already matches adaptive owner-gap counts.
The article-level role union provides no additional known hits on
this corpus.

Absolute coverage remains low: 22 of 160 requests yield a proof
under the hybrid joint-byline route. The annotations are catalog
topic tags and publisher contributor roles, rather than independent
judgments of expertise. The study has no unseen-catalog test.
Model inference produces embeddings for ranking only; verified
contributor roles come from the publisher XML. The table reports
matched reading work and does not include wall-clock times.

[Complete aggregates and paired intervals](results.json)

## Reproduce

Restore the 130 XMLs using the existing publisher manifest. Run on a
Slurm GPU compute allocation for the two retrieval views, or omit
`--embedding` for the lexical view alone.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.open_topic_role_bundles \
  benchmarks/plos-credit-cross-source-v1 NEW_RESULTS XML_DIRECTORY \
  --embedding QWEN_EMBEDDING
```
