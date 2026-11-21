# Verify an author-linked fallback from its topic text alone

Keep the original top-six source anchor and the
[author-posting fallback](../keyword-author-rescue-v1). The latter
proposes at most one source pair using a common publisher ORCID,
with one fused leader and the partner in the opposite top 24.
For a fallback pair, test a source-only reader before role
attribution. The reader sees each topic clue and the proposed
paper's indexed title and first abstract half. It receives no
author names, author link, request roles, or unindexed tail.
Use the resident open-weight Qwen3-Reranker-0.6B.

- **Weakest side:** score just the paper with the larger cosine
  gap to its dense leader.
- **Absolute:** score both selected papers in one two-comparison
  call and use the smaller matching score.
- **Relative:** also score the next best fused candidate on each
  side, excluding the selected paper. In a four-comparison call,
  use the smaller selected-minus-rival score difference.

In the earlier eLife development cohort, calibrate each cutoff
against all nominated pairs, including initially accepted and
fallback proposals. Deduplicate by source pair and availability
condition instead of using repeated role variants as independent
cases. Set the cutoff to the highest score of any incorrect
development proposal. Record retained correct proposals.
Apply the frozen cutoffs to fallback pairs only in the held
cohort. The existing top-six route remains common to all arms.
Report source-pair counts and strict proofs using the prototype
role band. Also measure strict proofs and wrong-source issuance
if the intended roles are supplied.

## Results

All verifier variants retain seven correct deduplicated development
keyword proposals and reject three wrong ones by the fitted cutoff.
The development policy has no new correct fallback with both
sources available. The cutoff models source validity among the
proposals, rather than assuming calibration had a sufficient count
of correct fallback cases. In the old excerpt arm the top-six
baseline itself has one wrong-source issuance when roles are supplied.
The fallback-only guards leave that result unchanged.

| Held keyword policy | Correct source pairs among 66 positives | Static strict proofs / 66 | Static adverse / 132 | Verified / 132 | Local P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-six anchor | 48 | 26 | 1 | 0 | 10.8–11.0 ms |
| Ungated author link | 61 | 33 | 2 | 0 | 10.8–11.0 ms |
| Verify weaker side | 53 | 29 | 1 | 26 | 19.2–19.4 ms |
| Verify both | 53 | 29 | 1 | 26 | 20.3–20.4 ms |
| Verify against rivals | 53 | 29 | 1 | 26 | 24.6–24.7 ms |

All three readers accept five of the 13 additionally reachable
keyword positives. One comparison gives the same result as two
or four in these measured conditions. The ungated fallback
has five oracle-role substitutions on the 204 development
missing-keyword views. Each reader reduces this count to zero.
No policy has a substitution on the 264 held missing views per
clue type. The three distinct wrong development proposals and
13 held pairs provide limited evidence for future risk. The
extra strict proofs are observed at higher query cost than
the anchor. The weakest-side route is the lower-cost verified
candidate for further independent testing.

Both tasks in Slurm array 46032 used ws3 RTX 5090 and match in
every non-timing field under reversed request order. Timings
include embedding, source ranking and the appropriate measured
reranker call. They exclude index preparation, free-form request
parsing and HTTP overhead. [Complete results](results.json)

## Reproduce

Restore both eLife projections from the pinned manifests and
run inside Slurm CUDA. Use a fresh output path and the model
revisions recorded in the results. Repeat with `--reverse`.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.credit.independent_source_confirmation \
  DEVELOPMENT_PROJECTION HELD_PROJECTION benchmarks/elife-prompt-holdout-v1/manifest.json \
  NISO_EXAMPLES benchmarks/plos-credit-edits-v1/role-training.json \
  benchmarks/engineering-cross-source-search-v1/results.json \
  QWEN_EMBEDDING QWEN_RERANKER NEW_RESULTS
```
