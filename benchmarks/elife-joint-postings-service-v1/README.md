# Test joint source recovery on a fresh eLife panel under load

Select 1,200 latest-version articles without replacement from the pinned
[publisher article tree](https://github.com/elifesciences/elife-article-xml/tree/58250dd64e27afae6bc2700bcddd84dd38776e0d/articles).
The stable hash selection excludes every article ID in the preceding
960-item sample before downloading or reading candidate content.
Reconstruct the earlier exposure inventory and check its archived hash.
Check each new and prior Git blob. Exclude new records sharing a DOI,
normalized title, normalized author name or valid ORCID with the union
of prior samples. Select one positive and one swapped-role control per
eligible linked article pair using the preceding construction. The
manifest fixes the data and family identities before service evaluation.

Use the frozen calibration and the same keyword and excerpt views as
the [preceding service study](../paired-role-prospective-service-v2).
The held models, thresholds and fallback come from earlier experiments.
Compare the single-slot baseline, the exact-agreement role route, the
linear-student-then-4B role cascade, and that same cascade with the
[joint source fallback](../paired-role-joint-postings-v1). The joint
fallback only runs if no paper pair has been selected and the verifier
forward remains unused. Require one dense role per clause, an indexed
same-person match across two papers, and source confirmation on both
clauses in a single forward. Charge the two source rows separately.

Serve all arms through one GPU worker and the same scheduled loopback
HTTP workload. Repeat at both Poisson arrival rates and rotate arm
order in two Slurm jobs. Report exact declared-role proof, observed
adverse issuance, role and source model work, and client-tail latency.
Compute descriptive strict-success intervals over connected paper and
recorded-author groups. Replays of the same requests contribute timing
and model work and do not increase the number of quality observations.

## Results

The two publisher-data builds produce byte-identical outputs.
The new sample excludes 135 non-research articles and 417 records
with prior author overlap. Another 29 fail the declaration or
abstract requirements. The 619 admitted documents yield 12 linked
paper pairs, forming 12 connected paper/recorded-author groups.
There are 48 unique requests, 12 positives and 12 role swaps per view.
The entity screen covers recorded ORCIDs and normalized names.

Four loopback runs from Slurm arrays 46251 and 46255 complete on
ws3 RTX 5090. All runs agree on every emitted evidence response.
The measured offered rates are 25.65 and 54.96 requests per second.

| Policy | Keyword strict / adverse | Excerpt strict / adverse | Source forwards / rows per 96 requests | Role forwards per 96 requests | Client P95 at 54.96 req/s (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-slot baseline | 8 / 0 | 10 / 0 | 2 / 2 | 28 | 127-134 |
| Exact-agreement 4B role route | 10 / 0 | 12 / 0 | 2 / 2 | 36 | 184-206 |
| Student then 4B | 10 / 0 | 12 / 0 | 2 / 2 | 32 | 144-161 |
| Cascade plus joint source fallback | 11 / 1 | 12 / 0 | 6 / 10 | 32 | 170-245 |

The student-first cascade reproduces the exact-agreement evidence
for all 48 requests. Its paired P95 reduction relative to that route
is 20%-25% across the four high-load replays. The single-slot baseline
remains faster and returns four fewer strict proofs across both views.

The additional source fallback gains one exact keyword proof and
introduces one adverse issuance with the correct source pair. Its
descriptive grouped interval for keyword improvement spans 0 to
25 percentage points relative to the student-first cascade. P95
increases in every high-load replay compared with that cascade.
The new panel supports the role cascade as the lower-cost route
at identical observed quality. It supports a narrower claim for joint
source recovery. Exact proof can improve, with additional verification,
tail latency and adverse outcomes observed in this workload. The
held structured sample remains small and measures a single-worker
loopback service. See the [frozen manifest](manifest.json) and
[all matched runs](results.json).

## Reproduce

Keep archived prior publisher snapshots, with the article tree in
`PUBLISHER_TREE`. Use the source inventory in the preceding manifest
and restore all earlier folders from the
[preceding data construction](../paired-role-prospective-service-v2#reproduce).
The optional `--fetch` flag fetches both recent samples from the pinned
publisher revision.

```sh
PYTHONPATH=src:. python -m experiments.credit.elife_joint_postings_holdout \
  PUBLISHER_TREE PREVIOUS_MANIFEST PREVIOUS_XML NEW_XML NEW_PROJECTION NEW_MANIFEST \
  EARLIER_ELIFE_XML EARLIER_PROMPT_XML ENGINEERING_DUAL_XML ENGINEERING_LINKED_XML --fetch

# Run inside a Slurm GPU allocation, and repeat with --reverse.
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.credit.paired_role_prospective_service \
  NEW_PROJECTION NEW_MANIFEST benchmarks/paired-role-verification-v1/results.json \
  NISO_EXAMPLES ROLE_TRAINING SCORE_BANDS SOURCE_FROZEN ROLE_FROZEN \
  QWEN_EMBEDDING QWEN_RERANKER QWEN_4B NEW_RESULTS \
  --joint-postings-service --plos PLOS_PROJECTION
```
