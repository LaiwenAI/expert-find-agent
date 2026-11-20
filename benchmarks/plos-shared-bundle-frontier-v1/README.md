# Share source reads across candidate experts and audit the read bound

Use 152 pinned PLOS publications, the publisher contribution
declarations and the 364 three-role conjunctions from the
[minimum-bundle benchmark](../plos-minimal-evidence-bundles-v1).
Both panels keep the same author order and return up to five
people with all roles supported by their own declarations.
Each output uses the smallest citation set, breaking ties by
DOI order. The linked panel selects 16 people with multiple
publications. The full panel contains 580, many with only one.

Materialize per-article author-role masks in local JSON files.
Each route has a query-local file memo and uses the same published
article-level role ceiling for unread declarations. Compare:

- **Owner frontier** completes one expert's optimistic proof at a time.
- **Shared sequential** keeps the first five still-possible experts
  and prioritizes the unresolved proof of the earliest one.
- **Shared Top-5** chooses the source shared by the most active
  optimistic proofs. Equal fan-out favors more required roles
  in the article ceiling, then DOI order.

A shared source read resolves every author in that file.
Only optimistic proof plans containing the read source need to
be recomputed. Revealing any other source can only weaken a
competing plan. A proof becomes exact when its optimistic
minimum citations have all been verified. Every result and
citation is checked against exhaustive publisher-grounded enumeration.

## Indispensable reads

For each file opened by the reference, reveal all other files
and keep its uncached roles unknown. Recompute the answer with
those masks set first to empty and then to their article ceiling.
A change under either admissible assignment means reading that
file is necessary to certify the exact output at the observed
state. Report the bound separately for the ordered identities,
their minimum citation counts, and the exact citations.
This treats the indexed article-role masks as upper bounds,
without additional cross-author constraints or source-side
information. The audit uses full labels retrospectively, never
as an input to the schedulers.

## Results

Both tasks of Slurm array `44993` on the `ws3` CPU allocation
reproduce every decision, file count and bound. Aggregate file
reads below sum the 364 requests within each panel.

| Panel, cache schedule | Files opened, all schedules | Necessary for identity order | Necessary for exact citations |
| --- | ---: | ---: | ---: |
| All authors, 0% | 8,625 | 8,612 | 8,618 |
| All authors, 50% | 4,011 | 4,005 | 4,011 |
| Linked authors, 0% | 5,357 | 4,979 | 5,138 |
| Linked authors, 50% | 2,848 | 2,751 | 2,837 |

Without cached declarations, the sequential reference is within
seven reads of the exact-output lower bound on the full panel.
Neither shared scheduler reduces file counts on any request.
The linked panel shows more room, including 132 reads required
by DOI tie-breaking rather than minimum citation count at zero
cache coverage. An equivalent-citation policy would require a
separate evaluation.

At zero cache coverage on the full panel, P95 is 237–239 µs
for the owner frontier, 261–264 µs for shared sequential, and
319–323 µs for shared Top-5. At full cache coverage, all routes
open zero files. Shared sequential runs in 17.6–17.8 µs versus
46.6–47.1 µs for the owner frontier in the full panel. Timing
uses median of five warm repetitions per request and measures
planning and local declaration decoding. It does not measure
raw publication fetches, inference, or end-to-end search latency.

These measurements support richer verified author-level index
metadata for reducing reads under exact-output requirements.
Cross-expert scheduling alone leaves little room under these
ceilings. The public projection contains publisher-declared
roles, and the ranking prior and role conjunctions are fixed.
[Aggregate results](results.json).

## Reproduce

Run on a Slurm CPU allocation. Use a fresh output file.

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m experiments.credit.shared_bundle_frontier \
  benchmarks/plos-credit-role-postings-v1/sources.json SCRATCH_DIRECTORY NEW_RESULTS
```
