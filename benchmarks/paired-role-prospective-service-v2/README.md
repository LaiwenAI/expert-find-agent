# Prospective paired-role verification under loopback load

Freeze the single-slot baseline, unconditional batched two-slot route
on a feasible unresolved query, and reserved batched rescue with
dense-label agreement from the
[development comparison](../paired-role-verification-v1).
Use the same shortlist rule, calibration result, model revisions
and source and role cutoffs. The executable checks their hashes.
Each request may use at most one source or role verification forward.
The role reader sees the two activity phrases and definitions.
Issuance requires matching declared roles for the same contributor
on both sources.

## Independent source snapshot

From the pinned [publisher XML revision](https://github.com/elifesciences/elife-article-xml/tree/58250dd64e27afae6bc2700bcddd84dd38776e0d),
select the highest XML version for each paper ID in [84000, 114000),
then choose 960 IDs by the hash of
`expert-paired-role-prospective-v2:<article-id>`. This is done before
reading the text and declarations. Check the Git object hashes.
Screen DOI, title, exact normalized author name and valid ORCID
against the earlier eLife, eLife parser and engineering snapshots.
Keep only indexed research articles with licensed XML, complete
recognizable declarations and at least three abstract sentences.
The manifest stores every sampled object's hash and disposition.
Distinct identities without normalized-name or ORCID overlap can
remain unrecognized.

Choose one positive and one role-swap control per eligible linked
paper pair. Use the same unindexed abstract fragment and indexed
keyword views as in development. These are structured test
requests with published declarations. The policies are fixed before
search on the new panel. Candidate papers, labels and proof scoring
are isolated from the request handler.

## Service replay

Use one GPU worker and 64 persistent loopback clients. Repeat the
two request views at two Poisson rates. Every arm receives the same
trace and scheduled arrivals. Rotate arm order, then rerun in reverse
order. Include queueing and transport in client P95 and report
achieved throughput, role-prompt tokens and total verifier forwards.
Score quality once per distinct request. Compute descriptive paired
strict-success intervals over connected source-paper and recorded
author groups, so repetition and linked paper pairs do not inflate
the nominal sample size.

## Results

Both jobs in Slurm array 46154 completed on RTX 5090. The catalog
admits 652 articles after the exposure and declaration filters.
Before declaration filtering, 165 papers were removed by the prior
author screen. The selected 27 pairs use 48 papers and form 22
connected paper/author groups. Both orders produce identical
evidence for every distinct request and complete all 216 requests
in every replay. Each view has 54 unique requests and 27 positives.

| Route | Keyword strict / adverse / issued | Excerpt strict / adverse / issued |
| --- | ---: | ---: |
| Single-slot baseline | 10 / 0 / 10 | 12 / 0 / 12 |
| Batched two-slot | 15 / 1 / 16 | 20 / 2 / 22 |
| Reserve batch and dense agreement | 12 / 0 / 12 | 18 / 0 / 18 |

Reserve batch with agreement gains 2/27 strict keyword positives and
6/27 strict excerpt positives. The descriptive grouped 95% intervals
for differences in strict success rate are [0.0, 18.5] and
[8.3, 34.5] percentage points, respectively. The batch-only route
recovers more positives with three adverse issuances across the
two views. The agreement route observed zero adverse issuances among
30 emitted results across both views. The two views share paper
pairs and do not provide a population risk guarantee.

| Actual arrival rate | Route | Client P95 (ms) | Achieved requests/s | Role forwards / 216 | Role prompt tokens / 216 |
| --- | --- | ---: | ---: | ---: | ---: |
| 29.0 | Single-slot baseline | 50.7–53.7 | 29.0 | 24 | 2314 |
| 29.0 | Batched two-slot | 83.9–84.9 | 28.9 | 118 | 18006 |
| 29.0 | Reserve batch and dense agreement | 79.6–82.6 | 28.9 | 118 | 16502 |
| 62.1 | Single-slot baseline | 96.9–104.9 | 61.2–61.3 | 24 | 2314 |
| 62.1 | Batched two-slot | 1297.0–1304.3 | 44.6–44.8 | 118 | 18006 |
| 62.1 | Reserve batch and dense agreement | 1193.1–1206.2 | 45.7–45.8 | 118 | 16502 |

At the higher rate, the reserve policy exceeds one second P95.
Server queueing contributes most of the tail. Both expanded routes
use 118 role forwards against 24 for the baseline and eight source
forwards each. The single-forward cap limits per-request dispatch
count while the larger share of requests entering the model saturates
this worker. Prioritize reducing rescue invocation and per-read cost
before testing a higher offered rate. These are single-worker
loopback results with structured conditions in a finite replay.
[Full results](results.json)

## Reproduce

Download each XML named in [the manifest](manifest.json) from the
pinned article directory and verify its Git blob ID. Restore the
earlier eLife role-nomination XML, eLife prompt-holdout XML, and
engineering dual-format and cross-source scout XML snapshots. These
four directories form the prior exposure inventory. Fetch the
pinned article Git tree and use
`experiments.credit.elife_paired_role_holdout` to rebuild the
projection and manifest. Check the projection hash before running
the benchmark.

```sh
PYTHONPATH=src:. python -m experiments.credit.elife_paired_role_holdout \
  ARTICLE_TREE NEW_XML_DIR NEW_PROJECTION NEW_MANIFEST \
  EARLIER_ELIFE_XML ELIFE_PROMPT_HOLDOUT_XML ENGINEERING_DUAL_XML ENGINEERING_LINKED_XML
```

Run inside a Slurm CUDA allocation using the frozen model paths
and source and role cutoffs. Repeat with a new output and
`--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.paired_role_prospective_service \
  NEW_PROJECTION NEW_MANIFEST benchmarks/paired-role-verification-v1/results.json \
  NISO_EXAMPLES ROLE_TRAINING SCORE_BANDS SOURCE_FROZEN ROLE_FROZEN \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_RESULTS
```
