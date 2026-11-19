# Source-specific contributor requirements across publications

This benchmark evaluates a closed set of published declarations. Each request supplies two source DOIs and asks for the same contributor to satisfy two roles in the first source and one role in the second. Generate two edit types. Replace the requested role in the second source, or swap the source assignments of two roles while keeping the role set fixed. A paired prediction is correct when both versions return the exact eligible contributor sets. The cases test source-specific role matching and same-contributor attribution before adding language parsing or document retrieval.

Seed the capture from a bounded [PLOS Search API](https://api.plos.org/solr/search-fields/) result for three engineering topics. Follow up with a bounded search on selected seed authors' publisher spellings. Verify the returned articles and source digests. Join identities across publications only when the publisher's JATS contributor fields have the same valid ORCID. Exclude the prior source inventory and reject authors whose ORCID or normalized byline name occurs in that inventory. The release includes publisher DOIs, XML hashes, dataset-local contributor identifiers, author ordinals and [CRediT](https://credit.niso.org/) roles. It does not include author names or copied publication text.

For each linked paper pair, hash-rank up to six role edits that change the gold answer, up to three stable role edits, and up to four source swaps that change the gold answer. The swap cohort deliberately challenges methods that ignore source scope. Direct labels are computed from owned roles in the requested article. No model outputs choose which edits are labeled. The comparison uses the same verified identity pool:

- Document union attributes article-level roles to all linked contributors.
- Owner union keeps roles attached to the individual and allows evidence from either article.
- Attributed scan requires every role to be assigned to that person in the specified source.
- Attributed postings intersect the source–role contributor sets.

The gold labels reflect publisher declarations in the captured papers. Missing declarations establish no fact about unobserved work, wider competence or enterprise suitability. Editions share paper pairs and contributors. Treat the rates as descriptive controlled tests.

## Results

The capture retains 25 PLOS ONE records after excluding 152 previously used DOIs and 217 previous publisher ORCIDs. The bounded follow-up checked 20 anchors and found two with linked additional sources. The scoped test covers six article pairs and 60 paired queries, including 18 label-changing role edits, 18 stable edits, and 24 source swaps. Repeat queries share a small number of same-contributor links.

| Method | Both versions exact / 60 | Source swaps exact / 24 | Unsupported attributions | Paired matching P95 |
| --- | ---: | ---: | ---: | ---: |
| Document union | 41 | 23 | 19 | 1.29 µs |
| Owner union | 35 | 0 | 25 | 1.58 µs |
| Attributed scan | 60 | 24 | 0 | 1.63 µs |
| Attributed postings | 60 | 24 | 0 | 1.77 µs |

Except for one role-edit family, owner union handles the original role edits and stable controls. It gives identical answers for both versions of every source swap because the aggregate role set never changes. The two scope-aware algorithms match the independently constructed publisher labels on all 120 query variants. Postings do not reduce latency on these small candidate pools. Timings cover two in-memory set lookups per measurement, after index construction.

The benchmark measures controlled failures when an explicit citation constraint is lost. The candidate IDs and sources are supplied, and query families overlap along the same publications. Results do not estimate the prevalence of these failures in unselected queries or expert recommendation accuracy under natural-language retrieval. [Capture manifest](capture.json) · [Fixture](fixture.json) · [Aggregate results](results.json)

## Reproduce

The [capture script](../../experiments/credit/harvest_scoped_credit.py) can recreate the workflow from a publisher capture folder and prior XML directories. Use the released fixture for exact evaluation, independent of publisher availability:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.source_scoped_credit evaluate FIXTURE_JSON NEW_RESULTS
```
