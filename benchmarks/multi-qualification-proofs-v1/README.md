# Bound same-person evidence assignments for three qualifications

Extend the [two-source projection](../role-pair-placement-v1) to
three topic-and-role conditions and query-specific source separation.
Use the same frozen 452-paper PLOS ONE and eLife catalog with verified
ORCID identity mapping. Compose three different topics and CRediT roles
from real declarations where a person can cite three different sources.
Select up to 12 hash-ordered triples per organizing person, then
deduplicate the requests. These are constructed positive requests
with overlapping sources and authors. They do not represent a random
sample of either publisher's queries.

Test two conditions on each request. The first requires all three
sources to differ. The second requires only the first and second
conditions to have distinct sources. A source may fill two permitted
conditions only if the same person has the requested role on that
source and the source is in each relevant retrieved pool. Known
matches additionally require the corresponding publisher topic
annotations. Missing annotations leave a match unjudged.

## Exact projection and shortlist bound

Use the same live BM25/dense reciprocal-rank pools at depth 128.
Compare a byline-only plan, exhaustive personal-role assignment,
catalog-projected role assignment, and a projected plan with an
additional per-query cap. The byline plan also uses the exact cap,
then verifies the chosen source declarations. When its ranked byline
pool contains no proof, it falls back to three topical sources.

For the all-distinct request, remove role postings for any person
present in fewer than three sources. For a single required distinct
pair, the floor is two. The packed sizes use the original source
headers and per-role bitmap format. Byline data, embeddings and
source records remain shared.

Within the plan, let `d` be the number of other conditions whose
sources must differ from a given condition's source. Retain its best
`d+1` ranked sources for a given person. If a winning assignment
used a lower-ranked source, at most `d` of the first `d+1` could
conflict with the other assigned sources. A replacement would not
increase the worst rank and would strictly improve the rank sum.
Thus the cap preserves the first result under the rank key
`(worst rank, rank sum, person, ordered sources)`. This argument
assumes strict facet rankings and source inequality as the only
interaction between individual choices. General ranked database
joins have related exact-enumeration methods.
[Optimal Algorithms for Ranked Enumeration](https://vldb.org/pvldb/vol13/p1582-tziavelis.pdf).

Read the chosen normalized declaration records and verify same-person
role support before returning. Compare exact outputs, source fetches,
and selection P95. Warm local request P95 includes live retrieval,
planning, declaration reads and serialization. The experiment does
not run a semantic topic verifier or measure service load.

## Results

Both tasks of Slurm array `45707` completed on the ws3 RTX 5090
and reproduced all non-timing outcomes. The projected and capped
plans match exhaustive role assignment on all 48 requests across
the two forms, including their selected person, ordered citations
and fetch plan.

| Query form | Full / projected role bytes | Enumerated candidates, exhaustive / projected / bounded | Planning P95, exhaustive → bounded | Known matches PLOS / eLife |
| --- | ---: | ---: | ---: | ---: |
| Three distinct sources | 6,114 / 68 | 469 / 93 / 93 | 0.189–0.289 → 0.049–0.050 ms | 9 / 8 |
| One distinct pair | 6,114 / 499 | 469 / 153 / 89 | 0.198–0.285 → 0.054–0.062 ms | 2 / 8 |

With three distinct sources the byline-only comparison finds
7 PLOS and 8 eLife known matches and reads 72 sources overall.
The role index finds 9 and 8 and reads 51. In that form,
projection does almost all the assignment reduction. With one
distinct pair, capping the isolated third condition to one source
reduces work beyond projection. Most request latency remains
in retrieval. Across the two publishers, P95 drops by roughly
0.16–0.24 ms from exhaustive to bounded planning on the all-distinct
form; the timing difference is for the complete local request.

Nine PLOS requests admit a known proof at depth 128 under either
source rule. The selected known counts fall from nine to two when
reuse is permitted, even though the role planner issues one more
proof. The more permissive structure can admit a higher-ranked role
assignment without full topic annotation. This demonstrates that
source-separation semantics and topic verification can change
selected evidence independently. It does not establish that an
unannotated topic is wrong or that requiring extra sources always
improves quality.

[Aggregates from both runs](results.json).

## Reproduce

Restore the frozen XML and manifest inputs from the two-source
benchmark. Use one Slurm GPU and repeat with `--order-offset 1`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false python -m experiments.credit.multi_qualification_proofs \
  PLOS_XML PLOS_FIXTURE ELIFE_XML ELIFE_MANIFEST QWEN_EMBEDDING \
  benchmarks/live-joint-source-request-v1/results.json NEW_RESULTS --depth 128
```
