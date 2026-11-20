# Stress-test author-role nomination when the index is incomplete

Use the frozen live PLOS and eLife heading inputs, query encoder, and
top-64 hybrid pools from the [published role-postings replay](../live-joint-role-postings-v1).
Each method opens three distinct original declarations.
Verification uses unchanged publisher records. The stress applies
only to the facts used to choose the sources.

For each coverage and corruption setting, three fixed SHA-256 masks
select fields from the same requests and sources. Missingness
withholds a source's personal role fields but preserves its contributor
identifiers. In the corruption condition a separate nominal 20%
hash cutoff selects eligible visible records and cyclically rotates
role sets between identified authors with nonempty role sets on the
same record. The per-article role union is unchanged. The sampled
and actually changed counts can differ. These are controlled conditions,
not estimates of missingness or attribution errors in enterprise data.

| Policy | Source plan |
| --- | --- |
| Joint byline | Pair complementary sources by the same contributor |
| Role priority | Prefer supported personal role matches and retain unknowns |
| Require two known roles | Use the role-priority plan only when both anchors share a contributor with the indexed roles |
| Require two known roles and a rank bound | Also cap the weaker anchor rank at the larger of eight and twice the byline plan's weaker anchor rank |

When a guarded rule declines the role plan, it reads the three joint
byline sources. The cap is fixed before the replay and uses no topic
labels. It is a retrieval-rank rule without a calibrated probability
interpretation. The joint byline control is computed once for each
request and reused as the matched reference across masks. Each
alternative verifies its own three declaration reads. This design does
not measure end-to-end serving latency.

## Results on RTX 5090

Both tasks of Slurm array `44580` ran on ws5 and reproduced all
nontiming outcomes. Each range is across the three deterministic masks
on the same panel. It is not a confidence interval. The clean complete
setting has identical masks. The byline control finds 49 of 160 PLOS
matches and 14 of 51 eLife matches. A known match requires both declared
roles and both publisher topic tags. Unlisted topics remain unresolved.

| Coverage and corruption | PLOS role priority | PLOS rank-bounded guard | eLife role priority | eLife rank-bounded guard |
| --- | ---: | ---: | ---: | ---: |
| 25%, clean | 44–62 | 43–49 | 14–22 | 14 |
| 50%, clean | 50–69 | 44–54 | 19–28 | 14–26 |
| 50%, role rotation | 42–62 | 44–55 | 19–26 | 14–26 |
| All available, clean | 82 | 67 | 28 | 27 |
| All available, role rotation | 75–82 | 64–68 | 20–28 | 27–28 |

At 50% clean coverage, role priority loses 4–16 PLOS requests that
the byline plan matched and gains 17–24 others. The rank bound
reduces the losses to 0–7, with 2–6 gains. At the byline control's
78 recommendations, role priority yields 41–54 known PLOS matches
and the rank-bounded guard 42–53, from a baseline of 49.
The eLife guard loses no known byline matches at 25% or 50% coverage
in the tested masks. This behavior does not reproduce in PLOS.

Under complete exposure and the rotation intervention, the PLOS
role-priority plan selects a first pair on 5–6 requests
where the pair appears supported in the index but lacks an original same-person,
two-role proof on that pair. The rank-bounded guard still selects four
such pairs per mask. This counts spent reads. The unchanged raw-source
verification will still require a valid proof before returning an
answer, and may find it with the third source. Requiring two indexed
roles alone is not a guarantee against assignment errors.

Planning alone has P95 of 0.130–0.133 ms for the PLOS byline plan
and 0.168–0.176 ms for the role view across masks. The eLife ranges
are 0.204–0.207 and 0.261–0.281 ms. The typed planning measure
includes masking-aware lookups and the guard check after the baseline
has already been prepared. It excludes index building, query encoding,
retrieval, source reads and answer verification.

## Reproduce

Restore the pinned publisher XML inventories and model. Use fresh
outputs for both runs and repeat with `--order-offset 1`. The aggregate
JSON stores one common copy of the replicated counts and both timing
records.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.robust_role_postings \
  PLOS_XML benchmarks/plos-credit-cross-source-v1 \
  ELIFE_XML benchmarks/elife-role-nomination-v1/manifest.json \
  QWEN_EMBEDDING benchmarks/live-joint-source-request-v1/results.json NEW_RESULTS
```

[Stress test results](results.json).
