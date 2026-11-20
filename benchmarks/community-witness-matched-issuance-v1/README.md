# Compare verification prompts at the same citation output size

The [2025 Cross Validated replay](../crossvalidated-held-window-routing-v1)
found more known witnesses with an answer-focused instruction and
also substantially more published citations. Keep the same 0.6B
verifier, candidate checks, request length, answer lead window, and
author grouping. Change only the verification instruction and the
issuance cutoff. Every reader obtains the strongest answer margin
per author, then ranks up to five authors for a request before the
issuance decision.

The evaluation includes the reused 2025 Cross Validated panel and
a new MathOverflow cohort of 2024–2025 questions linked to older
accepted answers. MathOverflow was captured through the same
Stack Exchange API. The release includes a manifest of public post
IDs and excludes post text and account profiles. Read original sources
at `https://mathoverflow.net/q/<question-id>` and
`https://mathoverflow.net/a/<answer-id>`.

The old and new communities share the pre-2023 content cutoff.
Within Cross Validated use the prior shared-source and title screen.
Before MathOverflow evaluation exclude exact normalized known-answer
text matches or at least 80% title-token overlap with either prior
Cross Validated panel. Numeric post identifiers are site-specific.

Compare four issuance rules:

1. Original instruction, score above zero.
2. Answer-focused instruction, score above zero.
3. Answer-focused instruction, cutoff chosen using 2024
   Cross Validated. Select the least restrictive strict threshold
   that emits no more citations than the original reader on
   development. Freeze the threshold for both test collections.
4. Answer-focused instruction, exactly the original total number
   of citations in each cohort. Rank all per-request eligible
   citations by score, tie-break by query position and answer
   index, and keep the original total. This post-hoc diagnostic
   uses scores and the original issuance count. It does not use
   target witness labels, but it requires scores for an entire
   cohort and is not an online policy.

Track known-answer hits, exact known citations, and total issued
citations and queries. Paired intervals keep connected components
of shared witness labels together. Non-target citations remain
unjudged. Equal output size controls one source of the differences
and does not establish true precision. Local verifier timing covers
model scoring and owner ranking, with retrieval already fixed.

## Results

Slurm job `43720` ran on ws5. The original and answer-focused
zero-threshold results on Cross Validated reproduce the previous
counts. The development panel has 68 queries and the two test
panels have 27 and 20 queries. MathOverflow supplies 328 older
answers in the candidate pool. One queried post linking directly
to the known answer was removed before the manifest was frozen.
The cross-site text screen removes no additional queries.

| Cohort | Original, hits / citations | Answer-focused at zero | Frozen cutoff | Exact citation count |
| --- | ---: | ---: | ---: | ---: |
| Cross Validated 2024 (development) | 38 / 58 | 45 / 75 | 39 / 58 | 39 / 58 |
| Cross Validated 2025 (reused) | 12 / 16 | 17 / 27 | 14 / 18 | 13 / 16 |
| MathOverflow (held out) | 13 / 13 | 17 / 18 | 11 / 11 | 13 / 13 |

Using the original development issuance count gives a frozen
cutoff of 1.25. It transfers imperfectly: Cross Validated emits
two additional citations, while MathOverflow emits two fewer.
At exact equal issuance, Cross Validated gains one query
(+3.70 points, descriptive paired 95% interval 0 to +11.54).
MathOverflow gains one query and loses one (0 points, −15 to +15).
At zero, both communities retain more known witnesses, alongside
an increase in issued citations. The larger benefit shrinks
substantially once the output count is held constant.

Original and answer-focused verifier P95 are 59.02 and 61.25 ms
for Cross Validated and 59.23 and 60.32 ms for MathOverflow.
The results do not show a cross-community equal-issuance gain.
The small cohorts and sparse positive-only labels leave an
independent assessment of the other issued citations necessary
for a precision claim. [Aggregate results](results.json)

## Reproduce

Use the existing 2024 and 2025 Cross Validated snapshots and
create the MathOverflow snapshot with the same chronological
content rule. The manifests pin source IDs and hashes.
The original API receipts are kept with the local snapshot.
Run inference inside a Slurm CUDA allocation.

```sh
PYTHONPATH=src:. python -m experiments.community.crossvalidated_witness \
  MATH_DIR --site mathoverflow --year 2024 --end-year 2025
PYTHONPATH=src:. PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.community.witness_matched_issuance \
    DEV_DIR/snapshot.json DEV_DIR/manifest.json \
    STATS_DIR/snapshot.json STATS_DIR/manifest.json \
    MATH_DIR/snapshot.json MATH_DIR/manifest.json \
    benchmarks/crossvalidated-held-window-routing-v1/results.json \
    QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS.json
```
