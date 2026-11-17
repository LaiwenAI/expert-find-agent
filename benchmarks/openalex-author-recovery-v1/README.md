# OpenAlex Temporal Author-Recovery Proxy

This is a small, publicly redistributable auxiliary benchmark, not a benchmark of enterprise suitability or actual individual contribution. It uses English-labeled publication-title metadata and author IDs from the [OpenAlex Works API](https://api.openalex.org/), whose [data are released under CC0](https://help.openalex.org/api/). The derived metadata and author-association labels are offered under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). No paper abstracts, researcher names, affiliations, private enterprise needs, or model responses are included.

Historical works from 2010--2019 form the evidence corpus. A 2020--2023 target title is a query; eligible known positives are its OpenAlex author IDs that also have at least one historical evidence record. Queries with no eligible author are excluded. An author association is not proof of a specific personal contribution, expertise, suitability, or ability to deliver a project. Authors omitted from the known-positive list are **unjudged**, not verified negatives. OpenAlex identities are machine-disambiguated and may contain errors.

Exact normalized target-title duplicates and historical titles with character four-gram Jaccard at least 0.8 against a target title are removed from historical evidence; duplicate target titles are collapsed. The 0.8 threshold was chosen after a preliminary audit of this snapshot. It removes 10 historical records but cannot detect all paraphrases or semantic leakage. Repeated authors across history and target are intentional. The source pool comes from five topical API searches and is neither representative nor exhaustive. Target-title retrieval is an author-recovery proxy, and evaluation is conditional on the authors present in the selected history pool; many target authors lack historical evidence in this snapshot.

## Files

- `experts.jsonl`: OpenAlex author IDs only.
- `evidence.jsonl`: historical work titles, years, owner IDs and work IDs; one row per work-author association.
- `queries.jsonl`: held-out work IDs, titles and publication years.
- `qrels.jsonl`: known eligible author IDs for each query; no negative labels.
- `source-records.jsonl`: filtered metadata used by the builder, including target works that were not eligible as queries.
- `manifest.json`: exact counts, API URLs, response hashes, timestamp and file hashes.

The API is dynamic: raw fetched responses are not redistributed here, so the stored response hashes alone do not make a future API request reproduce those bytes. The filtered source records and benchmark files freeze this release. To construct a new version, run `python -m experiments.openalex.fetch_snapshot` followed by `python -m experiments.openalex.build_author_recovery`; results may differ and are not a re-creation of this snapshot. Use the associated verifier and benchmark tools to validate this release and report known-author recall and reciprocal rank without treating unjudged authors as irrelevant.

```bash
PYTHONPATH=src python -m experiments.openalex.verify_author_recovery
PYTHONPATH=src python -m experiments.openalex.benchmark_author_recovery
PYTHONPATH=src python -m experiments.openalex.compare_candidate_pipeline --output artifacts/openalex-candidate-comparison
PYTHONPATH=src python -m experiments.openalex.audit_shortlist artifacts/openalex-candidate-comparison
```

The local exploratory lexical baseline returns a known author in the top ten for 21/99 queries with BM25-max and 27/99 with BM25-vote; known-author recall@10 is 0.202 and 0.256 respectively. Each query was timed once per method, and the experiment excludes initialization and model reasoning. These are baseline observations, not calibrated precision or statistical superiority. The 99 retained queries contain 126 known-positive links; 792 target works have no eligible author and are not benchmark queries. The manifest also records 3,472 target-author links without historical evidence in the selected pool. This selection excludes hard unavailable cases and must be reported with any result.

A separate exploratory comparison uses the same frozen corpus and rotates method order per query. Candidate language-model ranking followed by selective evidence lookup recovers a known author for 20/99 queries (known-author recall@10 0.190), versus 27/99 (0.256) for BM25-vote. Mean one-shot retrieval times on this local run were 0.81 ms and 5.77 ms respectively; candidate indexing additionally took 0.18 s after the shared 0.04 s lexical build. These observations exclude initialization and model calls, cover only a small corpus, and do not establish scalable latency or quality equivalence.

A subsequent post-hoc control adds two paths: precomputed BM25-vote, and a 50-candidate language-model shortlist reranked by the sum of up to five selective BM25 evidence scores. On the same 99 queries, their hits@10 are 27 and 25, with known-author recall@10 0.256 and 0.242. Precomputed BM25 matches the original BM25-vote ordered top ten for all 99 queries. In this later rotating-order run, mean one-shot times are 2.34 ms and 1.22 ms, respectively, excluding initialization and construction. This is exploratory reuse of the same data; the shortlist was selected after previous results, and the small corpus and one timing per query do not establish a general speed--accuracy frontier.

A post-hoc stage audit finds at least one known author in the top-50 shortlist for 31 of 99 queries and in the reranked top ten for 25. Of 27 queries where full BM25-vote returns a known author, four have no known author in the candidate shortlist. Two other queries hit after shortlist reranking despite a full BM25-vote miss. These are query-level comparisons: different known authors may account for hits at different stages. The known-positive list is incomplete and cannot establish which unjudged candidates are relevant.

A later replay from the source-release checkout verified all 99 queries and reproduced the published known-author metrics, including all 99 identical compiled-versus-original BM25-vote rankings. The run reused an existing Python environment; it does not verify a fresh dependency installation or extend the benchmark's relevance claims.

The reusable candidate pipeline also supports `aggregation='vote'`, summing the five highest BM25 source scores for each shortlisted expert. It omits zero-score experts, and the score can use more source records than are displayed. `python -m experiments.openalex.validate_vote_pipeline --output <new-directory>` compares this path with the previous evidence-fetch-and-vote implementation. A 99-query replay returned identical rankings, scores, and five displayed evidence items, with 25 known-author hits. Mean one-shot times were 1.07 and 0.92 ms, respectively; initialization and model reasoning were excluded. This is a reused small proxy, not a significance or industrial-quality claim.
